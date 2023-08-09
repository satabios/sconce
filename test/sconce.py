
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


class sconce:

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 64
        self.validate = True
        self.save = False
        self.goal = 'classficiation'
        self.expt_name = 'test-net'
        self.epochs = None
        self.learning_rate = 1e-4
        self.dense_model_valid_acc = 0

        self.fine_tune_epochs = 10
        self.fine_tune = False
        self.prune_model = True
        self.quantization = True
        self.num_finetune_epochs = 5
        self.best_sparse_model_checkpoint = dict()
        self.degradation_value = 1.2
        self.degradation_value_local = 1.2
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.callbacks = None
        self.sparsity_dict = None
        self.masks = dict()

        self.device = None


    def train(self) -> None:

            torch.cuda.empty_cache()

            self.model.to(self.device)

            val_acc = 0
            running_loss = 0.0


            epochs = self.epochs if self.fine_tune==False  else self.num_finetune_epochs
            for epoch in range(epochs):

                self.model.train()

                validation_acc = 0
                for data in tqdm(self.dataloader['train'], desc='train', leave=False):
                    # Move the data from CPU to GPU
                    if (self.goal != 'autoencoder'):
                        inputs, targets = data
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    elif (self.goal == 'autoencoder'):
                        inputs, targets = data.to(self.device), data.to(self.device)

                    # Reset the gradients (from the last iteration)
                    self.optimizer.zero_grad()

                    # Forward inference
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    # Backward propagation
                    loss.backward()

                    # Update optimizer and LR scheduler
                    self.optimizer.step()
                    if (self.scheduler is not None):
                        self.scheduler.step()

                    if (self.callbacks is not None):
                        for callback in self.callbacks:
                            callback()
                    running_loss += loss.item()


                running_loss = 0.0

                validation_acc = self.evaluate()
                if (validation_acc > val_acc):
                    print(f'Epoch:{epoch + 1} Train Loss: {running_loss / 2000:.5f} Validation Accuracy: {validation_acc:.5f}')
                    torch.save( copy.deepcopy(self.model.state_dict()), self.expt_name + '.pth')

    @torch.inference_mode()
    def evaluate(self, verbose=False):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.dataloader['test']:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            if (verbose):
                print('Test Accuracy: {} %'.format(acc))
            return acc

    ########## Model Profiling ##########
    def get_model_macs(self, inputs) -> int:
        global config
        return profile_macs(self.model, inputs)

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
        for param in self.model.parameters():
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
        for param in self.model.parameters():
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

        if (params.is_cuda):
            params = params.cpu().detach()
        if (torch.is_tensor(params)):
            params = params.item()

        size = params * data_width

        return size

    @torch.no_grad()
    def measure_latency(self, dummy_input, n_warmup=20, n_test=100):
        self.model.to("cpu")
        self.model.eval()
        # warmup
        for _ in range(n_warmup):
            _ = self.model(dummy_input)
        # real test
        t1 = time.time()
        for _ in range(n_test):
            _ = self.model(dummy_input)
        t2 = time.time()
        return round((t2 - t1) / n_test * 1000, 1)  # average latency in ms

    ########## Pruning ##########
    def plot_weight_distribution(self, bins=256, count_nonzero_only=False):
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        axes = axes.ravel()
        plot_index = 0
        for name, param in self.model.named_parameters():
            if param.dim() > 1:
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, bins=bins, density=True,
                            color='blue', alpha=0.5)
                else:
                    ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                            color='blue', alpha=0.5)
                ax.set_xlabel(name)
                ax.set_ylabel('density')
                plot_index += 1
        fig.suptitle('Histogram of Weights')
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()

    @torch.no_grad()
    def sensitivity_scan(self, dense_model_accuracy, scan_step=0.05, scan_start=0.1, scan_end=1.1, verbose=True):
        self.sparsity_dict = dict()
        sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
        accuracies = []
        named_conv_weights = [(name, param) for (name, param) \
                              in self.model.named_parameters() if param.dim() > 1]
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
                print(
                    f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]',
                    end='')
            accuracies.append(accuracy)

            self.degradation_value_local = self.degradation_value

            final_values_test = accuracies - np.asarray(self.dense_model_valid_acc)
            # print("Initial:", self.degradation_value_local)
            while(len(final_values_test[final_values_test>0])==0):
                # print("Updated:", self.degradation_value_local)
                self.degradation_value_local+= 0.5    #Small Increment to get a good degradation level
                if(self.degradation_value_local>15):  #Max Slack for Sparsity Value
                    break
                final_values_test = accuracies - np.asarray(self.dense_model_valid_acc)

            final_values = accuracies - np.asarray(self.dense_model_valid_acc)
            final_values_index_of_interest = np.where(final_values > 0, final_values < self.degradation_value_local, final_values)
            if (len(final_values_index_of_interest[final_values_index_of_interest == 1]) >= 1):
                selected_sparsity = sparsities[np.where(final_values_index_of_interest==1)[1][-1]]
                self.sparsity_dict[name] = selected_sparsity
            else:
                self.sparsity_dict[name] = 0.0

        if (verbose):
            lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
            fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
            axes = axes.ravel()
            plot_index = 0
            for name, param in self.model.named_parameters():
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

    def fine_grained_prune(self, tensor: torch.Tensor, sparsity: float) -> torch.Tensor:

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

        for name, param in self.model.named_parameters():
            if name in self.masks:
                param *= self.masks[name].to(self.device)

    # @staticmethod
    @torch.no_grad()
    def prune(self):
        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights

                self.masks[name] = self.fine_grained_prune(param, self.sparsity_dict[name])


    def compress(self, verbose=True) -> None:

        self.train()

        dense_model_size = self.get_model_size(count_nonzero_only=True)
        print(f"\nDense_model_size model after sensitivity size={dense_model_size / MiB:.2f} MiB")
        dense_validation_acc = self.evaluate(verbose=False)
        print("Original Model Validation Accuracy:", dense_validation_acc, "%")
        self.dense_model_valid_acc = dense_validation_acc

        self.sensitivity_scan(dense_model_accuracy=dense_validation_acc, verbose=False)

        self.prune()
        pruned_model_size = self.get_model_size(count_nonzero_only=True)
        validation_acc = self.evaluate(verbose=False)
        print(f"Pruned model has size={pruned_model_size / MiB:.2f} MiB = {pruned_model_size / dense_model_size * 100:.2f}% of Original model size")
        print(f"Sparsity for each Layer: {self.sparsity_dict}")
        self.callbacks = [lambda: self.apply()]
        self.fine_tune = True
        if (self.fine_tune):
            self.train()
        fine_tuned_pruned_model_size = self.get_model_size(count_nonzero_only=True)
        validation_acc = self.evaluate(verbose=False)
        if (verbose):
            print(
                f"Fine-Tuned Sparse model has size={fine_tuned_pruned_model_size / MiB:.2f} MiB = {fine_tuned_pruned_model_size / dense_model_size * 100:.2f}% of Original model size")
            print("Fine-Tuned Pruned Model Validation Accuracy:", validation_acc)

