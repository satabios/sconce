import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.optim as optim
import numpy as np
import random

import copy
import torch.optim as optim


import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

from torchprofile import profile_macs

assert torch.cuda.is_available(), \
"The current runtime does not have CUDA support." \
"Please go to menu bar (Runtime - Change runtime type) and select GPU"





random.seed(321)
np.random.seed(432)
torch.manual_seed(223)


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB



config = {
    'criterion' : nn.CrossEntropyLoss(),
    'batch_size': 64,
    'evaluate': True,
    'save':False,
    'goal':'classficiation',
    'expt_name':'test-net',
    'epochs':1,
    'learning_rate':1e-4,


    'fine_tune_epochs':2,
    'fine_tune':True,
    'prune':True,
    'quantization':True,
    'num_finetune_epochs':5,
    'best_sparse_model_checkpoint':dict(),

    'model':  None,
    'criterion':None,
    'optimizer':None,
    'scheduler':None,
    'dataloader':None,
    'callbacks':None,
    'sparsity_dict':None,
    'masks':dict()
    

    
}




class sconce():
    
    def __init__(self):
        
        global config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        
    def train(self) -> None:
        
        ## Add torch empty cache
        torch.cuda.empty_cache()
        
        
        config['model'].to(self.device)
        
        val_acc = 0
        running_loss = 0.0
        for epoch in range(config['epochs']):
            config['model'].train()
            
            validation_acc = 0
            for data in tqdm(config['dataloader']['train'], desc='train', leave=False):
                # Move the data from CPU to GPU
                if(config['goal'] != 'autoencoder'):
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                elif(config['goal'] == 'autoencoder'):
                    inputs, targets = data.to(self.device), data.to(self.device)

                # Reset the gradients (from the last iteration)
                config['optimizer'].zero_grad()

                # Forward inference
                outputs = config['model'](inputs)
                loss = config['criterion'](outputs, targets)

                # Backward propagation
                loss.backward()

                # Update optimizer and LR scheduler
                config['optimizer'].step()
                if(config['scheduler'] is not None):
                    config['scheduler'].step()

                if (config['callbacks'] is not None):
                    for callback in config['callbacks']:
                        callback()
                running_loss += loss.item()
                
            print(f'Epoch:{epoch + 1} Train Loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            
            validation_acc = self.evaluate()
            if(validation_acc> val_acc):
                torch.save(config['model'].state_dict(), config['expt_name']+'.pth')

    
    @torch.inference_mode()
    def evaluate(self):
        config['model'].to(self.device)
        config['model'].eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in config['dataloader']['test']:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = config['model'](images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

            print('Test Accuracy: {} %'.format(acc))
            return acc
        


    ########## Model Profiling ##########
    def get_model_macs(self, inputs) -> int:
            return profile_macs(config['model'], inputs)


    def get_sparsity(self, tensor: torch.Tensor) -> float:

        """
        calculate the sparsity of the given tensor
            sparsity = #zeros / #elements = 1 - #nonzeros / #elements
        """
        return 1 - float(tensor.count_nonzero()) / tensor.numel()


    def get_model_sparsity(self) -> float:
        """
        calculate the sparsity of the given model
            sparsity = #zeros / #elements = 1 - #nonzeros / #elements
        """
        num_nonzeros, num_elements = 0, 0
        for param in config['model'].parameters():
            num_nonzeros += param.count_nonzero()
            num_elements += param.numel()
        return 1 - float(num_nonzeros) / num_elements

    def get_num_parameters(self, count_nonzero_only=False) -> int:
        """
        calculate the total number of parameters of model
        :param count_nonzero_only: only count nonzero weights
        """
        num_counted_elements = 0
        for param in config['model'].parameters():
            if count_nonzero_only:
                num_counted_elements += param.count_nonzero()
            else:
                num_counted_elements += param.numel()
        return num_counted_elements


    def get_model_size(self, data_width=32, count_nonzero_only=False) -> int:
        """
        calculate the model size in bits
        :param data_width: #bits per element
        :param count_nonzero_only: only count nonzero weights
        """
        size = self.get_num_parameters(count_nonzero_only) * data_width
        if(size.is_cuda):
            size = size.cpu().detach()
        return size.item()
    
    @torch.no_grad()
    def measure_latency(self, dummy_input, n_warmup=20, n_test=100):
        config['model'].to("cpu")
        config['model'].eval()
        # warmup
        for _ in range(n_warmup):
            _ = config['model'](dummy_input)
        # real test
        t1 = time.time()
        for _ in range(n_test):
            _ = config['model'](dummy_input)
        t2 = time.time()
        return round((t2 - t1) / n_test* 1000, 1)  # average latency in ms

    ########## Pruning ##########
    def plot_weight_distribution(self, bins=256, count_nonzero_only=False):
        fig, axes = plt.subplots(3,3, figsize=(10, 6))
        axes = axes.ravel()
        plot_index = 0
        for name, param in config['model'].named_parameters():
            if param.dim() > 1:
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, bins=bins, density=True, 
                            color = 'blue', alpha = 0.5)
                else:
                    ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True, 
                            color = 'blue', alpha = 0.5)
                ax.set_xlabel(name)
                ax.set_ylabel('density')
                plot_index += 1
        fig.suptitle('Histogram of Weights')
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()



def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:

    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()


    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(num_elements * sparsity)
    # Step 2: calculate the importance of weight
    importance = tensor.abs()
    # Step 3: calculate the pruning threshold
    threshold = importance.view(-1).kthvalue(num_zeros).values
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = torch.gt(importance, threshold)

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask

def find_instance(name, obj, upto):
    if ( isinstance(obj, nn.Conv2d) or isinstance(obj, nn.Linear) ) :
        if(upto == "prune"):
            config['masks'][name] = fine_grained_prune(obj, config['sparsity_dict'][name]).to('cuda')
        elif(upto == "apply"):
            obj *= config['masks'][name]
        
    # elif isinstance(obj, list):                         #Find the use of list with names IDK fix this later
    #     for internal_obj in obj:
    #         find_instance(internal_obj)
    elif (hasattr(obj, '__class__')):
        for name, internal_obj in obj.named_children():
            find_instance(name, internal_obj)
    elif isinstance(obj, OrderedDict):
        for key, value in obj.items():
            find_instance(key, value)
    
class FineGrainedPruner():
    def __init__(self):
        global config
        # config['masks'] = self.prune(self)


   

    
    @torch.no_grad()
    def apply(self):
        find_instance("apply", obj, upto="apply")
        #updated
        # for name, param in config['model'].named_parameters():
        #     if name in config['masks']:
        #         param *= config['masks'][name]

    # @classmethod
    # @staticmethod
    @torch.no_grad()
    def prune(self):
        find_instance("prune", obj, upto="prune")
      
        # for name, param in config['model'].named_parameters():
        #     if param.dim() > 1: # we only prune conv and fc weights
        #         config['masks'][name] = self.fine_grained_prune(param, config['sparsity_dict'][name]).to('cuda')
        # return config['masks']

