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
from tqdm import tqdm
import gc
from torchprofile import profile_macs

from snntorch import utils
from collections import namedtuple
from fast_pytorch_kmeans import KMeans
from torch.nn import parameter
import ipdb
import snntorch
from snntorch import functional as SF

random.seed(321)
np.random.seed(432)
torch.manual_seed(223)

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

torch.cuda.synchronize()


class sconce:
    def __init__(self):
        """
        A class for training and evaluating neural networks with various optimization techniques.

        Attributes:
        - criterion: loss function used for training the model
        - batch_size: size of the batch used for training
        - validate: whether to validate the model during training
        - save: whether to save the model after training
        - goal: the goal of the model (e.g. classification)
        - experiment_name: name of the experiment
        - epochs: number of epochs for training
        - learning_rate: learning rate for the optimizer
        - dense_model_valid_acc: validation accuracy of the dense model
        - fine_tune_epochs: number of epochs for fine-tuning
        - fine_tune: whether to fine-tune the model
        - prune_model: whether to prune the model
        - prune_mode: mode of pruning (e.g. global, local)
        - quantization: whether to quantize the model
        - num_finetune_epochs: number of epochs for fine-tuning after pruning
        - best_sparse_model_checkpoint: checkpoint for the best sparse model
        - degradation_value: degradation value for pruning
        - degradation_value_local: local degradation value for pruning
        - model: the neural network model
        - criterion: loss function used for training the model
        - optimizer: optimizer used for training the model
        - scheduler: learning rate scheduler
        - dataloader: data loader for training and validation data
        - callbacks: callbacks for training the model
        - sparsity_dict: dictionary of sparsity values for each layer
        - masks: masks for pruning
        - Codebook: named tuple for codebook
        - codebook: codebook for quantization
        - channel_pruning_ratio: ratio of channels to prune
        - snn: whether to use spiking neural network
        - accuracy_function: function for calculating accuracy
        - bitwidth: bitwidth for quantization
        - device: device used for training the model
        """
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 64
        self.validate = True
        self.save = False
        self.goal = "classficiation"
        self.experiment_name = None
        self.epochs = None
        self.learning_rate = 1e-4
        self.dense_model_valid_acc = 0

        self.fine_tune_epochs = 10
        self.fine_tune = False
        self.prune_model = True
        self.prune_mode = ""
        self.quantization = True
        self.num_finetune_epochs = 5
        self.best_sparse_model_checkpoint = {}
        self.degradation_value = 1.2
        self.degradation_value_local = 1.2
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.callbacks = None
        self.sparsity_dict = None
        self.masks = {}
        self.comparison = True
        self.Codebook = namedtuple("Codebook", ["centroids", "labels"])
        self.codebook = None
        self.channel_pruning_ratio = None
        self.snn = False
        self.snn_num_steps = 50
        self.accuracy_function = None

        self.layer_of_interest = []
        self.conv_layer = []
        self.linear_layer = []
        self.handles = []
        self.temp_sparsity_list = []
        self.prune_indexes = []
        self.record_prune_indexes = False
        self.layer_idx = 0

        self.bitwidth = 4

        self.device = None

    def forward_pass_snn(self, data, mem_out_rec=None):
        """
        snn Forward Pass

        :param data: Input from the data loader
        :param mem_out_rec: Record Membrane Potential, if set to a value return both Membrane potential and Spikes
        :return: Return the Membrane Potential output of the network
        # :example:
        # .. jupyter-execute::
        #
        #   import sconce
        #   print(your_package_name.some_documented_func(1))
        """
        spk_rec = []
        mem_rec = []
        utils.reset(self.model)  # resets hidden states for all LIF neurons in net

        for step in range(self.snn_num_steps):  # data.size(0) = number of time steps
            spk_out, mem_out = self.model(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
            if mem_out_rec is not None:
                mem_rec.append(mem_out)
        if mem_out_rec is not None:
            return torch.stack(spk_rec), torch.stack(mem_rec)
        else:
            return torch.stack(spk_rec)

    def train(self) -> None:
        """
        Trains the model for a specified number of epochs using the specified dataloader and optimizer.
        If fine-tuning is enabled, the number of epochs is set to `num_finetune_epochs`.
        The function also saves the model state after each epoch if the validation accuracy improves.
        """

        torch.cuda.empty_cache()
        self.model.to(self.device)

        val_acc = 0
        running_loss = 0.0

        epochs = self.epochs if self.fine_tune == False else self.num_finetune_epochs
        for epoch in range(epochs):
            self.model.train()
            validation_acc = 0

            for i, data in enumerate(
                tqdm(self.dataloader["train"], desc="train", leave=False)
            ):
                # Move the data from CPU to GPU
                if self.goal != "autoencoder":
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                elif self.goal == "autoencoder":
                    inputs, targets = data.to(self.device), data.to(self.device)

                # Reset the gradients (from the last iteration)
                self.optimizer.zero_grad()

                # Forward inference
                if self.snn == True:
                    outputs = self.forward_pass_snn(inputs)
                    SF.accuracy_rate(outputs, targets) / 100

                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward propagation
                loss.backward()

                # Update optimizer and LR scheduler
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback()

                running_loss += loss.item()

            running_loss = 0.0

            validation_acc = self.evaluate()
            if validation_acc > val_acc:
                print(
                    f"Epoch:{epoch + 1} Train Loss: {running_loss / 2000:.5f} Validation Accuracy: {validation_acc:.5f}"
                )
                torch.save(
                    copy.deepcopy(self.model.state_dict()),
                    self.experiment_name + ".pth",
                )

    @torch.no_grad()
    def evaluate(self, Tqdm=True, verbose=False):
        """
        Evaluates the model on the test dataset and returns the accuracy.

        Args:
          verbose (bool): If True, prints the test accuracy.

        Returns:
          float: The test accuracy as a percentage.
        """

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            local_acc = []
            if Tqdm:
                loader = tqdm(self.dataloader["test"], desc="test", leave=False)
            else:
                loader = self.dataloader["test"]
            for i, data in enumerate(loader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                if self.prune_mode == "venum_sensitivity":
                    out = self.model(images)
                    total = len(images)
                    return
                elif self.snn:
                    outputs = self.forward_pass_snn(images, mem_out_rec=None)
                    correct += SF.accuracy_rate(outputs, labels) * outputs.size(1)
                    total += outputs.size(1)

                else:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0) - 1
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            if verbose:
                print("Test Accuracy: {} %".format(acc))
            return acc

    ########## Model Profiling ##########
    def get_model_macs(self, model, inputs) -> int:
        """
        Calculates the number of multiply-accumulate operations (MACs) required to run the given model with the given inputs.

        Args:
          model: The model to profile.
          inputs: The inputs to the model.

        Returns:
          The number of MACs required to run the model with the given inputs.
        """
        return profile_macs(model, inputs)

    def get_sparsity(self, tensor: torch.Tensor) -> float:
        """
        calculate the sparsity of the given tensor
            sparsity = #zeros / #elements = 1 - #nonzeros / #elements
        """
        return 1 - float(tensor.count_nonzero()) / tensor.numel()

    def get_model_sparsity(self, model: nn.Module) -> float:
        """

        Calculate the sparsity of the given PyTorch model.

        Sparsity is defined as the ratio of the number of zero-valued weights to the total number of weights in the model.
        This function iterates over all parameters in the model and counts the number of non-zero values and the total
        number of values.

        Args:
          model (nn.Module): The PyTorch model to calculate sparsity for.

        Returns:
          float: The sparsity of the model, defined as 1 - (# non-zero weights / # total weights).

        calculate the sparsity of the given model
            sparsity = #zeros / #elements = 1 - #nonzeros / #elements

        """
        num_nonzeros, num_elements = 0, 0
        for param in model.parameters():
            num_nonzeros += param.count_nonzero()
            num_elements += param.numel()
        return 1 - float(num_nonzeros) / num_elements

    def get_num_parameters(self, model: nn.Module, count_nonzero_only=False) -> int:
        """
        Calculate the total number of parameters of a PyTorch model.

        Args:
          model (nn.Module): The PyTorch model to count the parameters of.
          count_nonzero_only (bool, optional): Whether to count only the nonzero weights.
            Defaults to False.

        Returns:
          int: The total number of parameters of the model.
        """

        num_counted_elements = 0
        for param in model.parameters():
            if count_nonzero_only:
                num_counted_elements += param.count_nonzero()
            else:
                num_counted_elements += param.numel()
        return num_counted_elements

    def get_model_size(
        self, model: nn.Module, data_width=32, count_nonzero_only=False
    ) -> int:
        """
        calculate the model size in bits
        :param data_width: #bits per element
        :param count_nonzero_only: only count nonzero weights
        """
        return self.get_num_parameters(model, count_nonzero_only) * data_width

    @torch.no_grad()
    def measure_latency(self, model, dummy_input, n_warmup=20, n_test=100):
        """
        Measures the average latency of a given PyTorch model by running it on a dummy input multiple times.

        Args:
          model (nn.Module): The PyTorch model to measure the latency of.
          dummy_input (torch.Tensor): A dummy input to the model.
          n_warmup (int, optional): The number of warmup iterations to run before measuring the latency. Defaults to 20.
          n_test (int, optional): The number of iterations to run to measure the latency. Defaults to 100.

        Returns:
          float: The average latency of the model in milliseconds.
        """
        model = model.to("cpu")

        model.eval()

        dummy_input = dummy_input.to("cpu")

        if self.snn:
            if isinstance(model, nn.Sequential):
                for layer_id in range(len(model)):
                    layer = model[layer_id]
                    if isinstance((layer), snntorch._neurons.leaky.Leaky):
                        layer.mem = layer.mem.to("cpu")
            else:
                for module in model.modules():
                    if isinstance((module), snntorch._neurons.leaky.Leaky):
                        module.mem = module.mem.to("cpu")

        # warmup
        for _ in range(n_warmup):
            _ = model(dummy_input)
        # real test
        t1 = time.time()
        for _ in range(n_test):
            _ = model(dummy_input)
        t2 = time.time()
        return (t2 - t1) / n_test  # average latency in ms

    def plot_weight_distribution(self, bins=256, count_nonzero_only=False):
        """
        Plots the weight distribution of the model's named parameters.

        Args:
          bins (int): Number of bins to use in the histogram. Default is 256.
          count_nonzero_only (bool): If True, only non-zero weights will be plotted. Default is False.

        Returns:
          None
        """
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        axes = axes.ravel()
        plot_index = 0
        for name, param in self.model.named_parameters():
            if param.dim() > 1:
                ax = axes[plot_index]
                if count_nonzero_only:
                    param_cpu = param.detach().view(-1).cpu()
                    param_cpu = param_cpu[param_cpu != 0].view(-1)
                    ax.hist(param_cpu, bins=bins, density=True, color="blue", alpha=0.5)
                else:
                    ax.hist(
                        param.detach().view(-1).cpu(),
                        bins=bins,
                        density=True,
                        color="blue",
                        alpha=0.5,
                    )
                ax.set_xlabel(name)
                ax.set_ylabel("density")
                plot_index += 1
        fig.suptitle("Histogram of Weights")
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()

    @torch.no_grad()
    def sensitivity_scan(
        self,
        dense_model_accuracy,
        scan_step=0.05,
        scan_start=0.1,
        scan_end=1.0,
        verbose=True,
    ):
        """
        Scans the sensitivity of the model to weight pruning by gradually increasing the sparsity of each layer's weights
        and measuring the resulting accuracy. Returns a dictionary mapping layer names to the sparsity values that resulted
        in the highest accuracy for each layer.

        :param dense_model_accuracy: the accuracy of the original dense model
        :param scan_step: the step size for the sparsity scan
        :param scan_start: the starting sparsity for the scan
        :param scan_end: the ending sparsity for the scan
        :param verbose: whether to print progress information during the scan
        :return: a dictionary mapping layer names to the sparsity values that resulted in the highest accuracy for each layer
        """

        self.sparsity_dict = {}
        sparsities = np.flip(np.arange(start=scan_start, stop=scan_end, step=scan_step))
        accuracies = []
        named_conv_weights = [
            (name, param)
            for (name, param) in self.model.named_parameters()
            if param.dim() > 1
        ]
        original_model = copy.deepcopy(self.model)
        # original_dense_model_accuracy = self.evaluate()
        conv_layers = [
            module for module in self.model.modules() if (isinstance(module, nn.Conv2d))
        ]
        linear_layers = [
            module for module in self.model.modules() if (isinstance(module, nn.Linear))
        ]

        if self.prune_mode == "CWP":
            sorted_model = copy.deepcopy(self.apply_channel_sorting())
        layer_iter = tqdm(named_conv_weights, desc="layer", leave=False)
        for i_layer, (name, param) in enumerate(layer_iter):
            param_clone = param.detach().clone()
            accuracy = []
            desc = None
            if verbose:
                desc = f"scanning {i_layer}/{len(named_conv_weights)} weight - {name}"
                picker = tqdm(sparsities, desc)
            else:
                picker = sparsities
            hit_flag = False

            for sparsity in picker:
                if self.prune_mode == "venum_sensitivity":
                    # self.temp_sparsity_list[i_layer] = sparsity
                    self.layer_idx = i_layer
                    self.prune_mode = "venum"
                    self.find_instance(obj=self.model, sparsity=sparsity)
                    self.prune_mode = "venum_sensitivity"
                    self.evaluate(Tqdm=False)
                    for handle in self.handles:
                        handle.remove()
                    hit_flag = True
                if self.prune_mode == "GMP":
                    self.fine_grained_prune(param.detach(), sparsity=sparsity)
                    hit_flag = True
                elif (
                    self.prune_mode == "CWP"
                    and len(param.shape) > 2
                    and i_layer < (len(conv_layers) - 1)
                ):
                    # self.model = sorted_model
                    self.model = self.channel_prune_layerwise(
                        sorted_model, sparsity, i_layer
                    )
                    hit_flag = True
                ## TODO:
                ## Add conv CWP and linear CWP

                if hit_flag == True:
                    if self.prune_mode == "venum_sensitivity":
                        self.prune_mode = "venum"
                    acc = self.evaluate(Tqdm=False) - dense_model_accuracy
                    if self.prune_mode == "venum":
                        self.prune_mode = "venum_sensitivity"
                    if abs(acc) <= self.degradation_value:
                        self.sparsity_dict[name] = sparsity
                        break
                    elif sparsity == scan_start:
                        accuracy = np.asarray(accuracy)
                        best_possible_sparsity = sparsities[
                            np.where(accuracy == np.max(accuracy))[0][0]
                        ]
                        self.sparsity_dict[name] = best_possible_sparsity
                    else:
                        # restore
                        #
                        if self.prune_mode == "venum_sensitivity":
                            self.model = copy.deepcopy(original_model)
                        else:
                            param.copy_(param_clone)
                        accuracy.append(acc)
                        hit_flag = False
                # break

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
    def venum_apply(self):
        """
        Applies the Venum pruning algorithm to the model.

        This method iterates over the layers of the model and applies the Venum pruning algorithm
        to each layer based on the specified sparsity. It then evaluates the pruned model and
        updates the pruning mode accordingly.

        Returns:
            None
        """
        for layer_id, (_, sparsity) in enumerate(self.sparsity_dict.items()):
            self.layer_idx = layer_id
            self.find_instance(obj=self.model, sparsity=sparsity)

        self.prune_mode = "venum_sensitivity"
        self.evaluate(Tqdm=False)
        self.prune_mode = "venum"

        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def GMP_apply(self):
        """
        Applies the Group Masking Procedure (GMP) to the model's parameters.

        This function iterates over the model's named parameters and applies the corresponding mask
        if it exists in the `masks` dictionary. The mask is applied by element-wise multiplication
        with the parameter tensor.

        Args:
          self (object): The `sconce` object.

        Returns:
          None
        """
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param *= self.masks[name].to(self.device)

    # @staticmethod
    @torch.no_grad()
    def GMP_Pruning(self):
        """
        Applies Group-wise Magnitude Pruning (GMP) to the model's convolutional and fully-connected weights.
        The pruning is performed based on the sparsity levels specified in the `sparsity_dict` attribute.
        The pruned weights are stored in the `masks` attribute.
        """
        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                self.masks[name] = self.fine_grained_prune(
                    param, self.sparsity_dict[name]
                )

    def venum_prune(self, W, X, s, in_channel=0, kernel_size=0, cnn=False):
        """
        Prunes the weights in W based on the venum pruning metric.

        Args:
            W (torch.Tensor): The weight tensor.
            X (torch.Tensor): The input tensor.
            s (int): The sparsity ratio, indicating the percentage of weights to be pruned.
            in_channel (int, optional): The number of input channels. Default is 0.
            kernel_size (int, optional): The size of the kernel. Default is 0.
            cnn (bool, optional): Indicates whether the model is a CNN. Default is False.

        Returns:
            torch.Tensor: The pruned weight tensor.
        """
        metric = W.abs() * X.norm(p=2, dim=0)  # get the venum pruning metric
        _, sorted_idx = torch.sort(metric, dim=1)  # sort the weights per output

        if cnn:
            pruned_idx = sorted_idx[
                :, : int(in_channel * kernel_size[0] * kernel_size[1] * s)
            ]
        else:
            pruned_idx = sorted_idx[
                :, : int(in_channel * s)
            ]  # get the indices of the weights to be pruned

        if self.record_prune_indexes:
            self.prune_indexes.append(pruned_idx)
        with torch.no_grad():
            zeros_tensor = torch.zeros_like(W)
            # Use scatter_ to set the pruned indices to zero
            W.scatter_(dim=1, index=pruned_idx, src=zeros_tensor)
        if cnn:
            W = W.unflatten(dim=1, sizes=(in_channel, kernel_size[0], kernel_size[1]))

        return W

    def venum(self, sparstiy):
        def prune(module, inp, out):
            if isinstance(module, nn.Conv2d):
                in_channel = module.in_channels
                kernel_size = module.kernel_size
                weights = module.weight.flatten(1).clone()

                unfold = nn.Unfold(
                    module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                torch.cuda.empty_cache()
                inp_unfolded = unfold(inp[0])
                inp_unfolded = inp_unfolded.permute([1, 0, 2])
                inp_unfolded = inp_unfolded.flatten(1).T

                with torch.no_grad():
                    # print("cPre:",torch.count_nonzero(weights))
                    module.weight.data = self.venum_prune(
                        W=weights,
                        X=inp_unfolded,
                        s=sparstiy,
                        in_channel=in_channel,
                        kernel_size=kernel_size,
                        cnn=True,
                    )
                    # print("cPost:",torch.count_nonzero(module.weight.data))
                    # del inp_unfolded, weights

            elif isinstance(module, nn.Linear):
                weights = module.weight.data
                # print("LPre:",torch.count_nonzero(weights))
                module.weight.data = self.venum_prune(
                    W=weights, X=inp[0], s=sparstiy, cnn=False
                )
                # print("LPost:",torch.count_nonzero(module.weight.data))

            gc.collect()
            torch.cuda.empty_cache()

        return prune

    def find_instance(
        self, obj, object_of_importance=(nn.Conv2d, nn.Linear), sparsity=None
    ):
        """
        Recursively searches for instances of specified objects within a given object.

        Parameters:
        - obj: The object to search within.
        - object_of_importance: A tuple of classes representing the objects of interest. Default is (nn.Conv2d, nn.Linear).
        - sparsity: Optional parameter representing the sparsity value. Default is None.

        Returns:
        None
        """
        if isinstance(obj, object_of_importance):
            if self.prune_mode == "venum":
                if self.layer_idx == 0:
                    self.handles.append(obj.register_forward_hook(self.venum(sparsity)))
                elif self.layer_idx < 0:
                    return
                else:
                    self.layer_idx -= 1
            # Add Wanda and SparseGPT here
            else:
                if object_of_importance == nn.Conv2d:
                    self.conv_layer.append(obj)
                elif object_of_importance == nn.BatchNorm2d:
                    self.linear_layer.append(obj)
            return

        elif isinstance(obj, nn.Sequential):
            for layer_id in range(len(obj)):
                internal_obj = obj[layer_id]
                self.find_instance(internal_obj, object_of_importance, sparsity)
        elif isinstance(obj, list):
            for internal_obj in obj:
                self.find_instance(internal_obj, object_of_importance, sparsity)
        elif hasattr(obj, "__class__"):
            for internal_obj in obj.children():
                self.find_instance(internal_obj, object_of_importance, sparsity)
        elif isinstance(obj, OrderedDict):
            for key, value in obj.items():
                self.find_instance(value, object_of_importance, sparsity)

    def compress(self, verbose=True) -> None:
        """
        Compresses the neural network model using either Granular-Magnitude Pruning (GMP) or Channel-Wise Pruning (CWP).
        If GMP is used, the sensitivity of each layer is first scanned and then the Fine-Grained Pruning is applied.
        If CWP is used, the Channel-Wise Pruning is applied directly.
        After pruning, the model is fine-tuned using Stochastic Gradient Descent (SGD) optimizer with Cosine Annealing
        Learning Rate Scheduler.
        The original dense model and the pruned fine-tuned model are saved in separate files.
        Finally, the validation accuracy and the size of the pruned model are printed.

        Args:
          verbose (bool): If True, prints the validation accuracy and the size of the pruned model. Default is True.

        Returns:
          None
        """
        original_experiment_name = self.experiment_name
        if self.snn:
            original_dense_model = self.model

        else:
            original_dense_model = copy.deepcopy(self.model)
            torch.save(original_dense_model, "original_model.pth")
        torch.save(
            original_dense_model.state_dict(),
            self.experiment_name + "_original_weights.pth",
        )

        dense_model_size = self.get_model_size(
            model=self.model, count_nonzero_only=True
        )
        print(f"\nOriginal Dense Model Size Model={dense_model_size / MiB:.2f} MiB")
        dense_validation_acc = self.evaluate(verbose=False)
        print("Original Model Validation Accuracy:", dense_validation_acc, "%")
        self.dense_model_valid_acc = dense_validation_acc

        if self.prune_mode == "GMP":
            print("Granular-Magnitude Pruning")
            sensitivity_start_time = time.time()
            self.sensitivity_scan(
                dense_model_accuracy=dense_validation_acc, verbose=False
            )
            sensitivity_start_end = time.time() - sensitivity_start_time
            print("Sensitivity Scan Time(secs):", sensitivity_start_end)
            # self.sparsity_dict = {'0.weight': 0.6500000000000001, '3.weight': 0.5000000000000001, '7.weight': 0.7000000000000002}
            # self.sparsity_dict = {'backbone.conv0.weight': 0.20000000000000004, 'backbone.conv1.weight': 0.45000000000000007, 'backbone.conv2.weight': 0.25000000000000006, 'backbone.conv3.weight': 0.25000000000000006, 'backbone.conv4.weight': 0.25000000000000006, 'backbone.conv5.weight': 0.25000000000000006, 'backbone.conv6.weight': 0.3500000000000001, 'backbone.conv7.weight': 0.3500000000000001, 'classifier.weight': 0.7000000000000002}

            self.GMP_Pruning()  # FineGrained Pruning
            self.callbacks = [lambda: self.GMP_apply()]
            print(f"Sparsity for each Layer: {self.sparsity_dict}")
            self.fine_tune = True

        elif self.prune_mode == "CWP":
            print("\n Channel-Wise Pruning")
            sensitivity_start_time = time.time()
            self.sensitivity_scan(
                dense_model_accuracy=dense_validation_acc, verbose=False
            )
            sensitivity_start_end = time.time() - sensitivity_start_time
            print("Sensitivity Scan Time(secs):", sensitivity_start_end)

            # self.sparsity_dict = {'backbone.conv0.weight': 0.15000000000000002, 'backbone.conv1.weight': 0.15, 'backbone.conv2.weight': 0.15, 'backbone.conv3.weight': 0.15000000000000002, 'backbone.conv4.weight': 0.20000000000000004, 'backbone.conv5.weight': 0.20000000000000004, 'backbone.conv6.weight': 0.45000000000000007}
            print(f"Sparsity for each Layer: {self.sparsity_dict}")
            self.CWP_Pruning()  # Channelwise Pruning

            self.fine_tune = True
        elif self.prune_mode == "venum":
            print("\n Venum Pruning")
            sensitivity_start_time = time.time()
            self.prune_mode = "venum_sensitivity"
            self.sensitivity_scan(
                dense_model_accuracy=dense_validation_acc, verbose=False
            )
            sensitivity_start_end = time.time() - sensitivity_start_time
            print("Sensitivity Scan Time(secs):", sensitivity_start_end)
            self.prune_mode = "venum"
            # self.sparsity_dict = {'backbone.conv0.weight': 0.30000000000000004, 'backbone.conv1.weight': 0.45000000000000007, 'backbone.conv2.weight': 0.45000000000000007, 'backbone.conv3.weight': 0.5500000000000002, 'backbone.conv4.weight': 0.6000000000000002, 'backbone.conv5.weight': 0.7000000000000002, 'backbone.conv6.weight': 0.7500000000000002, 'backbone.conv7.weight': 0.8500000000000002, 'classifier.weight': 0.9500000000000003}
            print(f"Sparsity for each Layer: {self.sparsity_dict}")
            self.venum_apply()
            # self.fine_tune=True

            # self.callbacks = [lambda : self.venum_apply()]

        print(
            "Pruning Time Consumed (mins):", (time.time() - sensitivity_start_end) / 60
        )
        print(
            "Total Pruning Time Consumed (mins):",
            (time.time() - sensitivity_start_time) / 60,
        )

        pruned_model = copy.deepcopy(self.model)
        pruned_model_size = self.get_model_size(
            model=pruned_model, count_nonzero_only=True
        )
        pruned_validation_acc = self.evaluate(verbose=False)

        print(
            f"\nPruned Model has size={pruned_model_size / MiB:.2f} MiB(non-zeros) = {pruned_model_size / dense_model_size * 100:.2f}% of Original model size"
        )
        pruned_model_acc = self.evaluate()
        print(
            f"\nPruned Model has Accuracy={pruned_model_acc :.2f} MiB(non-zeros) = {pruned_model_acc - dense_validation_acc :.2f}% of Original model Accuracy"
        )

        if self.fine_tune:
            print("\n \n========== Fine-Tuning ==========")
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.num_finetune_epochs
            )
            self.experiment_name = self.experiment_name + "_pruned_" + "fine_tuning"
            self.train()
            if self.prune_mode == "venum":
                self.venum_apply()

            pruned_model = copy.deepcopy(self.model)

        fine_tuned_pruned_model_size = self.get_model_size(
            model=pruned_model, count_nonzero_only=True
        )
        fine_tuned_validation_acc = self.evaluate(verbose=False)

        torch.save(
            pruned_model.state_dict(),
            self.experiment_name + "_pruned_fine_tuned_weights" + ".pth",
        )
        if self.snn == False:
            torch.save(pruned_model, "pruned_model.pth")
        accuracies = [
            dense_validation_acc,
            pruned_validation_acc,
            fine_tuned_validation_acc,
        ]
        self.compare_models(original_dense_model, pruned_model, accuracies)

        if verbose:
            print(
                f"Fine-Tuned Sparse model has size={fine_tuned_pruned_model_size / MiB:.2f} MiB = {fine_tuned_pruned_model_size / dense_model_size * 100:.2f}% of Original model size"
            )
            print(
                "Fine-Tuned Pruned Model Validation Accuracy:",
                fine_tuned_validation_acc,
            )

    # def k_means_quantize(self, fp32_tensor: torch.Tensor, bitwidth=4,codebook=None):
    #
    #     """
    #     quantize tensor using k-means clustering
    #     :param fp32_tensor:
    #     :param bitwidth: [int] quantization bit width, default=4
    #     :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    #     :return:
    #         [Codebook = (centroids, labels)]
    #             centroids: [torch.(cuda.)FloatTensor] the cluster centroids
    #             labels: [torch.(cuda.)LongTensor] cluster label tensor
    #     """
    #
    #     if codebook is None:
    #       # get number of clusters based on the quantization precision
    #       # hint: one line of code
    #       n_clusters = 1 << self.bitwidth
    #
    #       # use k-means to get the quantization centroids
    #       kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
    #       labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
    #       centroids = kmeans.centroids.to(torch.float).view(-1)
    #       self.codebook = self.Codebook(centroids, labels)
    #
    #     # decode the codebook into k-means quantized tensor for inference
    #     # hint: one line of code
    #     # ipdb.set_trace()
    #     quantized_tensor = self.codebook.centroids[self.codebook.labels]
    #
    #     fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    #     return codebook
    #
    #
    #
    # @torch.no_grad()
    # def kmeansquantize(self):
    #   """
    #   Applies k-means quantization to the model's parameters and returns a dictionary of codebooks.
    #   If the bitwidth is a dictionary, applies quantization to the named parameters in the dictionary.
    #   Otherwise, applies quantization to all parameters with more than one dimension.
    #   Returns a dictionary of codebooks, where each key is the name of a parameter and each value is a codebook.
    #   """
    #   codebook = {}
    #   if isinstance(self.bitwidth, dict):
    #     for name, param in self.model.named_parameters():
    #       if name in bitwidth:
    #         codebook[name] = self.k_means_quantize(param, bitwidth=self.bitwidth[name])
    #   else:
    #     for name, param in self.model.named_parameters():
    #       if param.dim() > 1:
    #         codebook[name] = self.k_means_quantize(param, bitwidth=self.bitwidth)
    #   return codebook

    #
    # @torch.no_grad()
    # def apply_quantization(self, update_centroids):
    #     """
    #     Applies quantization to the model's parameters using k-means clustering.
    #
    #     Args:
    #       update_centroids (bool): Whether to update the codebook centroids.
    #
    #     Returns:
    #       None
    #     """
    #     self.codebook = self.quantize()
    #     for name, param in self.model.named_parameters():
    #       if name in self.codebook:
    #         if update_centroids:
    #           update_codebook(param, codebook=self.codebook[name])
    #         self.codebook[name] = k_means_quantize(
    #           param, codebook=self.codebook[name])

    def compare_models(self, original_dense_model, pruned_fine_tuned_model, accuracies):
        """
        Compares the performance of two PyTorch models: an original dense model and a pruned and fine-tuned model.
        Prints a table of metrics including latency, MACs, and model size for both models and their reduction ratios.

        Args:
        - original_dense_model: a PyTorch model object representing the original dense model
        - pruned_fine_tuned_model: a PyTorch model object representing the pruned and fine-tuned model

        Returns: None
        """
        # accuracies = accuracies
        input_shape = list(next(iter(self.dataloader["test"]))[0].size())
        input_shape[0] = 1
        dummy_input = torch.randn(input_shape).to("cpu")
        pruned_fine_tuned_model = pruned_fine_tuned_model.to("cpu")

        # Parse through snn model and send to cpu
        if self.snn:
            for model in [original_dense_model, pruned_fine_tuned_model]:
                if isinstance(model, nn.Sequential):
                    for layer_id in range(len(original_dense_model)):
                        layer = original_dense_model[layer_id]
                        if isinstance((layer), snntorch._neurons.leaky.Leaky):
                            layer.mem = layer.mem.to("cpu")
                else:
                    for module in model.modules():
                        if isinstance((layer), snntorch._neurons.leaky.Leaky):
                            layer.mem = layer.mem.to("cpu")

        original_dense_model = original_dense_model.to("cpu")
        original_latency = self.measure_latency(
            model=original_dense_model, dummy_input=dummy_input
        )
        pruned_latency = self.measure_latency(
            model=pruned_fine_tuned_model, dummy_input=dummy_input
        )

        print("\n ................. Comparison Table  .................")
        table_template = "{:<15} {:<15} {:<15} {:<15}"
        print(table_template.format("", "Original", "Pruned", "Reduction Ratio"))
        print(
            table_template.format(
                "Latency (ms)",
                round(original_latency * 1000, 1),
                round(pruned_latency * 1000, 1),
                round(original_latency / pruned_latency, 1),
            )
        )

        # 2. measure the computation (MACs)
        original_macs = self.get_model_macs(original_dense_model, dummy_input)
        pruned_macs = self.get_model_macs(pruned_fine_tuned_model, dummy_input)
        table_template = "{:<15} {:<15} {:<15} {:<15}"
        print(
            table_template.format(
                "MACs (M)",
                round(original_macs / 1e6),
                round(pruned_macs / 1e6),
                round(original_macs / pruned_macs, 1),
            )
        )

        # 3. measure the model size (params)
        original_param = self.get_num_parameters(
            original_dense_model, count_nonzero_only=True
        ).item()
        pruned_param = self.get_num_parameters(
            pruned_fine_tuned_model, count_nonzero_only=True
        ).item()
        print(
            table_template.format(
                "Param (M)",
                round(original_param / 1e6, 2),
                round(pruned_param / 1e6, 2),
                round(original_param / pruned_param, 1),
            )
        )

        # 4. Accuracies

        print(
            table_template.format(
                "Accuracies (%)",
                round(accuracies[0], 3),
                round(accuracies[-1], 3),
                str(round(accuracies[-1] - accuracies[0], 3)),
            )
        )

        # put model back to cuda
        # pruned_model = pruned_fine_tuned_model.to("cuda")
        # model = original_dense_model.to("cuda")

    def CWP_Pruning(self):
        """
        Applies channel pruning to the model using the specified channel pruning ratio.
        Returns the pruned model.
        """
        sorted_model = self.apply_channel_sorting()
        # self.model = self.channel_prune(sorted_model, self.channel_pruning_ratio)
        self.sparsity_dict = [value for key, value in self.sparsity_dict.items()]
        self.model = self.channel_prune(sorted_model, self.sparsity_dict)

    def get_input_channel_importance(self, weight):
        """
        Computes the importance of each input channel in a weight tensor.

        Args:
          weight (torch.Tensor): The weight tensor to compute channel importance for.

        Returns:
          torch.Tensor: A tensor containing the importance of each input channel.
        """

        in_channels = weight.shape[1]
        importances = []
        # compute the importance for each input channel
        for i_c in range(weight.shape[1]):
            channel_weight = weight.detach()[:, i_c]
            ##################### YOUR CODE STARTS HERE #####################
            importance = torch.norm(channel_weight)
            ##################### YOUR CODE ENDS HERE #####################
            importances.append(importance.view(1))
        return torch.cat(importances)

    @torch.no_grad()
    def apply_channel_sorting(self):
        """
        Applies channel sorting to the model's convolutional and batch normalization layers.
        Returns a copy of the model with sorted channels.

        Returns:
        model (torch.nn.Module): A copy of the model with sorted channels.
        """

        model = copy.deepcopy(self.model)  # do not modify the original model
        # fetch all the conv and bn layers from the backbone

        all_convs = []
        all_bns = []

        # Universal Layer Seeking by Parsing
        def find_instance(obj, object_of_importance):
            if isinstance(obj, object_of_importance):
                if object_of_importance == nn.Conv2d:
                    all_convs.append(obj)
                elif object_of_importance == nn.BatchNorm2d:
                    all_bns.append(obj)
                return None
            elif isinstance(obj, list):
                for internal_obj in obj:
                    find_instance(internal_obj, object_of_importance)
            elif hasattr(obj, "__class__"):
                for internal_obj in obj.children():
                    find_instance(internal_obj, object_of_importance)
            elif isinstance(obj, OrderedDict):
                for key, value in obj.items():
                    find_instance(value, object_of_importance)

        find_instance(obj=model, object_of_importance=nn.Conv2d)
        find_instance(obj=model, object_of_importance=nn.BatchNorm2d)

        # # iterate through conv layers
        # for i_conv in range(len(all_convs) - 1):
        #     # each channel sorting index, we need to apply it to:
        #     # - the output dimension of the previous conv
        #     # - the previous BN layer
        #     # - the input dimension of the next conv (we compute importance here)
        #     prev_conv = all_convs[i_conv]
        #     prev_bn = all_bns[i_conv]
        #     next_conv = all_convs[i_conv + 1]
        #     # note that we always compute the importance according to input channels
        #     importance = self.get_input_channel_importance(next_conv.weight)
        #     # sorting from large to small
        #     sort_idx = torch.argsort(importance, descending=True)
        #
        #     # apply to previous conv and its following bn
        #     prev_conv.weight.copy_(
        #         torch.index_select(prev_conv.weight.detach(), 0, sort_idx)
        #     )
        #     for tensor_name in ["weight", "bias", "running_mean", "running_var"]:
        #         tensor_to_apply = getattr(prev_bn, tensor_name)
        #         tensor_to_apply.copy_(
        #             torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
        #         )
        #
        #     # apply to the next conv input (hint: one line of code)
        #     ##################### YOUR CODE STARTS HERE #####################
        #     next_conv.weight.copy_(
        #         torch.index_select(next_conv.weight.detach(), 1, sort_idx)
        #     )

        return model

    def get_num_channels_to_keep(self, channels: int, prune_ratio: float) -> int:
        """A function to calculate the number of layers to PRESERVE after pruning
        Note that preserve_rate = 1. - prune_ratio
        """
        ##################### YOUR CODE STARTS HERE #####################
        return int(round(channels * (1.0 - prune_ratio)))
        ##################### YOUR CODE ENDS HERE #####################

    def get_num_channels_to_keep(self, channels: int, prune_ratio: float) -> int:
        """A function to calculate the number of layers to PRESERVE after pruning
        Note that preserve_rate = 1. - prune_ratio
        """
        ##################### YOUR CODE STARTS HERE #####################
        return int(round(channels * (1.0 - prune_ratio)))
        ##################### YOUR CODE ENDS HERE #####################

    @torch.no_grad()
    def channel_prune_layerwise(
        self, model: nn.Module, prune_ratio: Union[List, float], i_layer
    ) -> nn.Module:
        """Apply channel pruning to each of the conv layer in the backbone
        Note that for prune_ratio, we can either provide a floating-point number,
        indicating that we use a uniform pruning rate for all layers, or a list of
        numbers to indicate per-layer pruning rate.
        """
        # sanity check of provided prune_ratio
        assert isinstance(prune_ratio, (float, list))

        all_convs = []
        all_bns = []

        # Universal Layer Seeking by Parsing
        def find_instance(obj, object_of_importance):
            if isinstance(obj, object_of_importance):
                if object_of_importance == nn.Conv2d:
                    all_convs.append(obj)
                elif object_of_importance == nn.BatchNorm2d:
                    all_bns.append(obj)
                return None
            elif isinstance(obj, list):
                for internal_obj in obj:
                    find_instance(internal_obj, object_of_importance)
            elif hasattr(obj, "__class__"):
                for internal_obj in obj.children():
                    find_instance(internal_obj, object_of_importance)
            elif isinstance(obj, OrderedDict):
                for key, value in obj.items():
                    find_instance(value, object_of_importance)

        # we prune the convs in the backbone with a uniform ratio
        new_model = copy.deepcopy(model)  # prevent overwrite
        find_instance(obj=new_model, object_of_importance=nn.Conv2d)
        find_instance(obj=new_model, object_of_importance=nn.BatchNorm2d)
        n_conv = len(all_convs)
        # note that for the ratios, it affects the previous conv output and next
        # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...

        # we only apply pruning to the backbone features

        # apply pruning. we naively keep the first k channels
        # assert len(all_convs) == len(all_bns)
        # for i_ratio, p_ratio in enumerate(prune_ratio):
        prev_conv = all_convs[i_layer]
        if self.snn == False:
            prev_bn = all_bns[i_layer]
        next_conv = all_convs[i_layer + 1]
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels
        n_keep = self.get_num_channels_to_keep(original_channels, prune_ratio)

        # prune the output of the previous conv and bn
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        if self.snn == False:
            prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
            prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
            prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
            prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

        # prune the input of the next conv (hint: just one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
        ##################### YOUR CODE ENDS HERE #####################

        return new_model

    @torch.no_grad()
    def channel_prune(
        self, model: nn.Module, prune_ratio: Union[List, float]
    ) -> nn.Module:
        """Apply channel pruning to each of the conv layer in the backbone
        Note that for prune_ratio, we can either provide a floating-point number,
        indicating that we use a uniform pruning rate for all layers, or a list of
        numbers to indicate per-layer pruning rate.
        """
        # sanity check of provided prune_ratio
        assert isinstance(prune_ratio, (float, list))

        all_convs = []
        all_bns = []

        # Universal Layer Seeking by Parsing
        def find_instance(obj, object_of_importance):
            if isinstance(obj, object_of_importance):
                if object_of_importance == nn.Conv2d:
                    all_convs.append(obj)
                elif object_of_importance == nn.BatchNorm2d:
                    all_bns.append(obj)
                return None
            elif isinstance(obj, list):
                for internal_obj in obj:
                    find_instance(internal_obj, object_of_importance)
            elif hasattr(obj, "__class__"):
                for internal_obj in obj.children():
                    find_instance(internal_obj, object_of_importance)
            elif isinstance(obj, OrderedDict):
                for key, value in obj.items():
                    find_instance(value, object_of_importance)

        # we prune the convs in the backbone with a uniform ratio
        new_model = copy.deepcopy(model)  # prevent overwrite
        find_instance(obj=new_model, object_of_importance=nn.Conv2d)
        find_instance(obj=new_model, object_of_importance=nn.BatchNorm2d)
        n_conv = len(all_convs)
        # note that for the ratios, it affects the previous conv output and next
        # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
        if isinstance(prune_ratio, list):
            assert len(prune_ratio) == n_conv - 1
        else:  # convert float to list
            prune_ratio = [prune_ratio] * (n_conv - 1)

        # we only apply pruning to the backbone features

        # apply pruning. we naively keep the first k channels
        assert len(all_convs) == len(all_bns)
        for i_ratio, p_ratio in enumerate(prune_ratio):
            prev_conv = all_convs[i_ratio]
            prev_bn = all_bns[i_ratio]
            next_conv = all_convs[i_ratio + 1]
            original_channels = prev_conv.out_channels  # same as next_conv.in_channels
            n_keep = self.get_num_channels_to_keep(original_channels, p_ratio)

            # prune the output of the previous conv and bn
            prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
            prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
            prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
            prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
            prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

            # prune the input of the next conv (hint: just one line of code)
            ##################### YOUR CODE STARTS HERE #####################
            next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
            ##################### YOUR CODE ENDS HERE #####################

        return new_model
