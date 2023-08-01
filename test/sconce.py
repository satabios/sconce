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
    'degradation_value' : 1.0,

    'model':  None,
    'criterion':None,
    'optimizer':None,
    'scheduler':None,
    'dataloader':None,
    'callbacks':None,
    'sparsity_dict':None,
    'masks':dict(),
    'iter_test' : [],

    'device' : None
    

    
}


class sconce:
    
    def __init__(self):
        
        global config

       
        
    def train_prune(self, verbose=True) -> None:
 
        self.train()
        validation_acc = self.evaluate(verbose =False)

        dense_model_size = self.get_model_size( count_nonzero_only=True)
        if(verbose):
            print(f"Original model has size={dense_model_size / MiB:.2f} MiB")

        self.sensitivity_scan(validation_acc, verbose=False)
        self.prune()

        pruned_model_size = self.get_model_size(count_nonzero_only=True)
        if(verbose):
            print(f"Pruned model has size={pruned_model_size / MiB:.2f} MiB")

        config['callbacks'] = [lambda: self.apply()]
        if(config['fine_tune']):
            self.train()

        fine_tuned_pruned_model_size = self.get_model_size(count_nonzero_only=True)
        if(verbose):
            print(f"Fine-Tuned Sparse model has size={fine_tuned_pruned_model_size / MiB:.2f} MiB = {fine_tuned_pruned_model_size / dense_model_size * 100:.2f}% of Original model size")

    def train(self) -> None:
        
        ## Add torch empty cache
        torch.cuda.empty_cache()
        
        
        config['model'].to(config['device'])
        
        val_acc = 0
        running_loss = 0.0
        for epoch in range(config['epochs']):
            config['model'].train()
            
            validation_acc = 0
            for data in tqdm(config['dataloader']['train'], desc='train', leave=False):
                # Move the data from CPU to GPU
                if(config['goal'] != 'autoencoder'):
                    inputs, targets = data
                    inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                elif(config['goal'] == 'autoencoder'):
                    inputs, targets = data.to(config['device']), data.to(config['device'])

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
    def evaluate(self, verbose=False):
        config['model'].to(config['device'])
        config['model'].eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in config['dataloader']['test']:
                images, labels = images.to(config['device']), labels.to(config['device'])
                outputs = config['model'](images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            if(verbose):
                print('Test Accuracy: {} %'.format(acc))
            return acc
        


    ########## Model Profiling ##########
    def get_model_macs(self, inputs) -> int:
            global config
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
        global config
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
        global config
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
        global config
        params = self.get_num_parameters(count_nonzero_only) 

        if(params.is_cuda):
            params = params.cpu().detach()
        if(torch.is_tensor(params)):
            params = params.item()
            

        size = params* data_width

        return size
    
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


    @torch.no_grad()
    def sensitivity_scan(self, dense_model_accuracy, scan_step=0.1, scan_start=0.3, scan_end=1.1, verbose=True):
        config['sparsity_dict'] = dict()
        sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
        accuracies = []
        named_conv_weights = [(name, param) for (name, param) \
                            in config['model'].named_parameters() if param.dim() > 1]
        for i_layer, (name, param) in enumerate(named_conv_weights):
            param_clone = param.detach().clone()
            accuracy = []
            desc = None
            if (verbose):
                desc = f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'
                picker = tqdm(sparsities, desc)
            else:
                picker = sparsities
            for sparsity in picker:
                self.fine_grained_prune(param.detach(), sparsity=sparsity)
                acc = self.evaluate()
                if verbose:
                    print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
                # restore
                param.copy_(param_clone)
                accuracy.append(acc)
            if verbose:
                print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
            accuracies.append(accuracy)
            
            final_values = accuracies - np.asarray(config['degradation_value'])
            final_values_index_of_interest = np.where(final_values>0, final_values<config['degradation_value'], final_values)
            if(len(final_values_index_of_interest[final_values_index_of_interest == 1])>1):
                selected_sparsity =  sparsities[final_values_index_of_interest[final_values_index_of_interest == 1][-1]]
                config['sparsity_dict'][name] = selected_sparsity
            else:
                config['sparsity_dict'][name] = 0.0


    
        if(verbose):
            lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
            fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)),figsize=(15,8))
            axes = axes.ravel()
            plot_index = 0
            for name, param in config['model'].named_parameters():
                if param.dim() > 1:
                    ax = axes[plot_index]
                    curve = ax.plot(sparsities, accuracies[plot_index])
                    line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
                    ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
                    ax.set_ylim(80, 95)
                    ax.set_title(name)
                    ax.set_xlabel('sparsity')
                    ax.set_ylabel('top-1 accuracy')
                    ax.legend([
                        'accuracy after pruning',
                        f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
                    ])
                    ax.grid(axis='x')
                    plot_index += 1
            fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
            fig.tight_layout()
            fig.subplots_adjust(top=0.925)
            plt.show()
    
    def fine_grained_prune(self, tensor: torch.Tensor, sparsity : float) -> torch.Tensor:

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
    
    @torch.no_grad()
    def apply(self):

        for name, param in config['model'].named_parameters():
            if name in config['masks']:
                param *= config['masks'][name].to(config['device'])

    # @staticmethod
    @torch.no_grad()
    def prune(self):
        for name, param in config['model'].named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights

                config['masks'][name] = self.fine_grained_prune(param, config['sparsity_dict'][name])



class FineGrainedPruner:
    def __init__(self):
        global config

    def fine_grained_prune(self, tensor: torch.Tensor, sparsity : float) -> torch.Tensor:

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
    
    @torch.no_grad()
    def apply(self):

        for name, param in config['model'].named_parameters():
            if name in config['masks']:
                param *= config['masks'][name].to(config['device'])

    # @staticmethod
    @torch.no_grad()
    def prune(self):
        for name, param in config['model'].named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights

                config['masks'][name] = self.fine_grained_prune(param, config['sparsity_dict'][name])

    


