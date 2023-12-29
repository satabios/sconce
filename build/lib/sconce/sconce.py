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

from thop import profile
from prettytable import PrettyTable
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
        self.params = []
        self.qat_config = "x86"

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
        self.venum_sorted_list = []
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

    def train(self, model=None) -> None:
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
    def venum_evaluate(self, Tqdm=True, verbose=False):
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

            loader = self.dataloader["test"]
            for i, data in enumerate(loader):
                images, labels = next(iter(loader))
                images, labels = images.to(self.device), labels.to(self.device)

                out = self.model(images)
                total = len(images)

            return

    @torch.no_grad()
    def evaluate(self, model=None, device=None, Tqdm=True, verbose=False):
        """
        Evaluates the model on the test dataset and returns the accuracy.

        Args:
          verbose (bool): If True, prints the test accuracy.

        Returns:
          float: The test accuracy as a percentage.
        """
        if model != None:
            self.model = model
        if device != None:
            final_device = device
        else:
            final_device = self.device

        self.model.to(final_device)
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
                images, labels = images.to(final_device), labels.to(final_device)
                # if ( "venum" in self.prune_mode ):
                #     out = self.model(images)
                #     total = len(images)
                #     return
                if self.snn:
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

    def measure_inference_latency(
        self, model, device, input_data, num_samples=100, num_warmups=10
    ):
        model.to(device)
        model.eval()

        x = input_data.to(device)

        with torch.no_grad():
            for _ in range(num_warmups):
                _ = model(x)
        torch.cuda.synchronize()

        with torch.no_grad():
            start_time = time.time()
            for _ in range(num_samples):
                _ = model(x)
                torch.cuda.synchronize()
            end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_ave = elapsed_time / num_samples

        return elapsed_time_ave

    def save_torchscript_model(self, model, model_dir, model_filename):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_filepath = os.path.join(model_dir, model_filename)

        torch.jit.save(torch.jit.script(model), model_filepath)

    def load_torchscript_model(self, model_filepath, device):
        model = torch.jit.load(model_filepath, map_location=device)

        return model

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

    def get_model_size_weights(self, mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        mdl_size = round(os.path.getsize("tmp.pt") / 1e6, 3)
        os.remove("tmp.pt")
        return mdl_size

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
        torch.cuda.synchronize()
        # real test
        t1 = time.time()
        for _ in range(n_test):
            _ = model(dummy_input)
            torch.cuda.synchronize()
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
        named_all_weights = [
            (name, param)
            for (name, param) in self.model.named_parameters()
            if param.dim() > 1
        ]
        named_conv_weights = [
            (name, param)
            for (name, param) in self.model.named_parameters()
            if param.dim() > 2
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
            sortd = self.apply_channel_sorting()
            sorted_model = copy.deepcopy(sortd)

        if "venum" in self.prune_mode:
            if self.prune_mode == "venum-cwp":
                named_all_weights = named_conv_weights

            list_of_sparsities = [0] * (len(named_all_weights) - 1)
            sparsity_dict = {count: ele for count, ele in enumerate(list_of_sparsities)}
            self.venum_apply(sparsity_dict)

        layer_iter = tqdm(named_all_weights, desc="layer", leave=False)
        original_prune_mode = self.prune_mode
        for i_layer, (name, param) in enumerate(layer_iter):
            param_clone = param.detach().clone()
            accuracy = []
            desc = None
            if verbose:
                desc = f"scanning {i_layer}/{len(named_all_weights)} weight - {name}"
                picker = tqdm(sparsities, desc)
            else:
                picker = sparsities
            hit_flag = False

            for sparsity in picker:
                if (
                    "venum" in self.prune_mode
                    and len(param.shape) > 2
                    and i_layer < (len(conv_layers) - 1)
                ):
                    # self.temp_sparsity_list[i_layer] = sparsity
                    self.layer_idx = i_layer
                    self.prune_mode = original_prune_mode

                    list_of_sparsities = [0] * (len(layer_iter) - 1)
                    list_of_sparsities[i_layer] = sparsity
                    sparsity_dict = {
                        count: ele for count, ele in enumerate(list_of_sparsities)
                    }
                    if self.prune_mode == "venum-cwp":
                        self.venum_CWP_Pruning(original_model, sparsity_dict)
                    else:
                        self.venum_apply(sparsity_dict)

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
                    # if self.prune_mode == "venum_sensitivity":
                    #     self.prune_mode = original_prune_mode
                    acc = self.evaluate(Tqdm=False) - dense_model_accuracy
                    # if ("venum" in self.prune_mode):
                    #     self.prune_mode = "venum_sensitivity"
                    if abs(acc) <= self.degradation_value:
                        self.sparsity_dict[name] = sparsity
                        self.model = copy.deepcopy(original_model)
                        break
                    elif sparsity == scan_start:
                        accuracy = np.asarray(accuracy)
                        best_possible_sparsity = sparsities[
                            np.where(accuracy == np.max(accuracy))[0][0]
                        ]
                        self.sparsity_dict[name] = best_possible_sparsity
                        self.model = copy.deepcopy(original_model)
                    else:
                        # restore
                        #
                        if "venum" in self.prune_mode:
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
    def venum_apply(self, sparsity_dict):
        for layer_id, (_, sparsity) in enumerate(sparsity_dict.items()):
            if sparsity > 0:
                self.layer_idx = layer_id
                self.find_instance(obj=self.model, sparsity=sparsity)

        self.venum_evaluate(Tqdm=False)
        # self.prune_mode = "venum"

        for handle in self.handles:
            handle.remove()

    # @torch.no_grad()
    # def venum_apply(self, sparsity_dict):
    #
    #     for layer_id, (_, sparsity) in enumerate(sparsity_dict.items()):
    #         self.layer_idx = layer_id
    #         self.find_instance(obj=self.model, sparsity=sparsity)
    #
    #     self.prune_mode = "venum_sensitivity"
    #     self.venum_evaluate(Tqdm=False)
    #     self.prune_mode = "venum"
    #
    #     for handle in self.handles:
    #         handle.remove()

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
        metric = W.abs() * X.norm(p=2, dim=0)  # get the venum pruning metric

        if self.prune_mode == "venum_cwp" and cnn == True:
            norm = torch.norm(metric, dim=1)
            _, sorted_idx = torch.sort(norm)
            self.venum_sorted_list.append(sorted_idx)

        else:
            # Venum GMP

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
                W = W.unflatten(
                    dim=1, sizes=(in_channel, kernel_size[0], kernel_size[1])
                )

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
                    if self.prune_mode == "venum_cwp":
                        self.venum_prune(
                            W=weights,
                            X=inp_unfolded,
                            s=sparstiy,
                            in_channel=in_channel,
                            kernel_size=kernel_size,
                            cnn=True,
                        )
                    else:
                        self.venum_prune(
                            W=weights,
                            X=inp_unfolded,
                            s=sparstiy,
                            in_channel=in_channel,
                            kernel_size=kernel_size,
                            cnn=True,
                        )

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
        if isinstance(obj, object_of_importance):
            if "venum" in self.prune_mode:
                if self.layer_idx == 0:
                    # print("LID, sp:", obj, self.layer_idx, sparsity)
                    self.handles.append(obj.register_forward_hook(self.venum(sparsity)))
                    self.layer_idx -= 1
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

        input_shape = list(next(iter(self.dataloader["test"]))[0].size())
        input_shape[0] = 1

        current_device = next(original_dense_model.parameters()).device
        dummy_input = torch.randn(input_shape).to(current_device)

        self.params.append(
            [
                self.evaluate(model=original_dense_model),
                self.measure_latency(
                    model=original_dense_model, dummy_input=dummy_input
                ),
                self.get_num_parameters(model=original_dense_model),
                self.get_model_size(
                    model=original_dense_model, count_nonzero_only=True
                ),
            ]
        )
        save_file_name = self.experiment_name + "_original.pt"
        self.save_torchscript_model(
            model=original_dense_model, model_dir="./", model_filename=save_file_name
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
            print("Sensitivity Scan Time(mins):", sensitivity_start_end / 60)
            # self.sparsity_dict = {'0.weight': 0.6500000000000001, '3.weight': 0.5000000000000001, '7.weight': 0.7000000000000002}
            # self.sparsity_dict = {'backbone.conv0.weight': 0.20000000000000004, 'backbone.conv1.weight': 0.45000000000000007, 'backbone.conv2.weight': 0.25000000000000006, 'backbone.conv3.weight': 0.25000000000000006, 'backbone.conv4.weight': 0.25000000000000006, 'backbone.conv5.weight': 0.25000000000000006, 'backbone.conv6.weight': 0.3500000000000001, 'backbone.conv7.weight': 0.3500000000000001, 'classifier.weight': 0.7000000000000002}

            self.GMP_Pruning()  # FineGrained Pruning
            self.callbacks = [lambda: self.GMP_apply()]
            print(f"Sparsity for each Layer: {self.sparsity_dict}")
            self.fine_tune = True

        elif self.prune_mode == "CWP":
            print("\n Channel-Wise Pruning")
            sensitivity_start_time = time.time()
            # self.sensitivity_scan(
            #     dense_model_accuracy=dense_validation_acc, verbose=False
            # )
            sensitivity_start_end = time.time() - sensitivity_start_time
            print("Sensitivity Scan Time(mins):", sensitivity_start_end / 60)

            self.sparsity_dict = {
                "backbone.conv0.weight": 0.15000000000000002,
                "backbone.conv1.weight": 0.15,
                "backbone.conv2.weight": 0.15,
                "backbone.conv3.weight": 0.15000000000000002,
                "backbone.conv4.weight": 0.20000000000000004,
                "backbone.conv5.weight": 0.20000000000000004,
                "backbone.conv6.weight": 0.45000000000000007,
            }
            print(f"Sparsity for each Layer: {self.sparsity_dict}")
            self.CWP_Pruning()  # Channelwise Pruning
            self.fine_tune = True

        # elif self.prune_mode == "venum":
        #     print("\n Venum Pruning")
        #     sensitivity_start_time = time.time()
        #     self.prune_mode = "venum_sensitivity"
        #     self.sensitivity_scan(dense_model_accuracy= dense_validation_acc, verbose=False)
        #     sensitivity_start_end = time.time() - sensitivity_start_time
        #     print("Sensitivity Scan Time(secs):", sensitivity_start_end)
        #     self.prune_mode = "venum"
        #     # self.sparsity_dict = {'backbone.conv0.weight': 0.30000000000000004, 'backbone.conv1.weight': 0.45000000000000007, 'backbone.conv2.weight': 0.45000000000000007, 'backbone.conv3.weight': 0.5500000000000002, 'backbone.conv4.weight': 0.6000000000000002, 'backbone.conv5.weight': 0.7000000000000002, 'backbone.conv6.weight': 0.7500000000000002, 'backbone.conv7.weight': 0.8500000000000002, 'classifier.weight': 0.9500000000000003}
        #     print(f"Sparsity for each Layer: {self.sparsity_dict}")
        #     self.venum_apply(self.sparsity_dict)
        #     self.fine_tune=True

        elif "venum" in self.prune_mode:
            print("\n Venum CWP Pruning")
            sensitivity_start_time = time.time()
            self.sensitivity_scan(
                dense_model_accuracy=dense_validation_acc, verbose=False
            )
            sensitivity_start_end = time.time() - sensitivity_start_time
            print("Sensitivity Scan Time(mins):", sensitivity_start_end / 60)

            # self.sparsity_dict = {'backbone.conv0.weight': 0.3500000000000001, 'backbone.conv1.weight': 0.15000000000000002, 'backbone.conv2.weight': 0.1, 'backbone.conv3.weight': 0.15000000000000002, 'backbone.conv4.weight': 0.20, 'backbone.conv5.weight': 0.20, 'backbone.conv6.weight': 0.30000000000000004}
            print(f"Sparsity for each Layer: {self.sparsity_dict}")
            self.venum_apply(self.sparsity_dict)
            if self.prune_mode == "venum-cwp":
                self.venum_CWP_Pruning(original_dense_model, self.sparsity_dict)
            self.fine_tune = True

        print(
            "Pruning Time Consumed (mins):",
            (time.time() - sensitivity_start_end / 60) / 60,
        )
        print(
            "Total Pruning Time Consumed (mins):",
            (time.time() - sensitivity_start_time) / 60,
        )

        pruned_model = copy.deepcopy(self.model)

        current_device = next(pruned_model.parameters()).device
        dummy_input = torch.randn(input_shape).to(current_device)

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
            print("\n \n==================== Fine-Tuning ====================")
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.num_finetune_epochs
            )

            self.train()
            save_file_name = self.experiment_name + "_pruned_fine_tuned" + ".pt"
            self.save_torchscript_model(
                model=self.model, model_dir="./", model_filename=save_file_name
            )

            if self.prune_mode == "venum":
                self.venum_apply(self.sparsity_dict)

            pruned_model = copy.deepcopy(self.model)

        fine_tuned_pruned_model_size = self.get_model_size(
            model=pruned_model, count_nonzero_only=True
        )
        fine_tuned_validation_acc = self.evaluate(verbose=False)

        if verbose:
            print(
                f"Fine-Tuned Sparse model has size={fine_tuned_pruned_model_size / MiB:.2f} MiB = {fine_tuned_pruned_model_size / dense_model_size * 100:.2f}% of Original model size"
            )
            print(
                "Fine-Tuned Pruned Model Validation Accuracy:",
                fine_tuned_validation_acc,
            )

        quantized_model, model_fp32_trained = self.qat()

        model_list = [original_dense_model, pruned_model, quantized_model]

        self.compare_models(model_list=model_list)

    def evaluate_model(self, model, test_loader, device, criterion=None):
        model.eval()
        model.to(device)

        running_loss = 0
        running_corrects = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels).item()
            else:
                loss = 0

            # statistics
            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        eval_loss = running_loss / len(test_loader.dataset)
        eval_accuracy = running_corrects / len(test_loader.dataset)

        return eval_loss, eval_accuracy

    def print_model_size(self, mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
        os.remove("tmp.pt")

    def qat(self):
        print(
            "\n \n==================== Quantization-Aware Training(QAT) ===================="
        )

        def get_all_layers(model, parent_name=""):
            layers = []
            for name, module in model.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                layers.append((full_name, module))
                if isinstance(module, nn.Module):
                    layers.extend(get_all_layers(module, parent_name=full_name))
            return layers

        fusing_layers = [
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.batchnorm.BatchNorm1d,
        ]

        def detect_sequences(lst):
            detected_sequences = []

            i = 0
            while i < len(lst):
                if i + 2 < len(lst) and [type(l) for l in lst[i : i + 3]] == [
                    fusing_layers[0],
                    fusing_layers[1],
                    fusing_layers[2],
                ]:
                    detected_sequences.append(
                        np.take(name_list, [i for i in range(i, i + 3)]).tolist()
                    )
                    i += 3
                elif i + 1 < len(lst) and [type(l) for l in lst[i : i + 2]] == [
                    fusing_layers[0],
                    fusing_layers[1],
                ]:
                    detected_sequences.append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                # if i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[0], fusing_layers[2]]:
                #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
                #     i += 2
                # elif i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[1], fusing_layers[2]]:
                #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
                #     i += 2
                elif i + 1 < len(lst) and [type(l) for l in lst[i : i + 2]] == [
                    fusing_layers[3],
                    fusing_layers[2],
                ]:
                    detected_sequences.append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                elif i + 1 < len(lst) and [type(l) for l in lst[i : i + 2]] == [
                    fusing_layers[3],
                    fusing_layers[4],
                ]:
                    detected_sequences.append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                else:
                    i += 1

            return detected_sequences

        original_model = copy.deepcopy(self.model)

        model_fp32 = copy.deepcopy(self.model)
        model_fp32 = nn.Sequential(
            torch.quantization.QuantStub(), model_fp32, torch.quantization.DeQuantStub()
        )

        model_fp32.eval()

        all_layers = get_all_layers(model_fp32)
        name_list = []
        layer_list = []
        for name, module in all_layers:
            name_list.append(name)
            layer_list.append(module)

        fusion_layers = detect_sequences(layer_list)

        model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig(
            self.qat_config
        )

        # fuse the activations to preceding layers, where applicable
        # this needs to be done manually depending on the model architecture
        model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, fusion_layers)

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model needs to be set to train for QAT logic to work
        # the model that will observe weight and activation tensors during calibration.
        model_fp32_prepared = torch.ao.quantization.prepare_qat(
            model_fp32_fused.train()
        )
        self.model = model_fp32_prepared
        self.train()

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        model_fp32_trained = copy.deepcopy(self.model)
        model_fp32_trained.to("cpu")
        model_fp32_trained.eval()

        model_int8 = torch.ao.quantization.convert(model_fp32_trained, inplace=True)
        model_int8.eval()
        # # torch.save(model_int8, 'quantized_model.pt')
        # # torch.save(
        # #     model_int8.state_dict(),
        # #     self.experiment_name + "_quantized" + ".pth",
        # # # )
        # input_shape = list(next(iter(self.dataloader["test"]))[0].size())
        # input_shape[0] = 1
        # current_device = "cpu"
        # dummy_input = torch.randn(input_shape).to(current_device)
        # #
        # # self.params.append([self.evaluate(model=model_int8),
        # #                     self.measure_latency(model=model_int8, dummy_input=dummy_input),
        # #                     self.get_num_parameters(model=model_int8),
        # #                     self.get_model_size(model=model_int8, count_nonzero_only=True)])
        #
        # save_file_name = self.experiment_name + "_int8.pt"
        # self.save_torchscript_model(model=model_int8, model_dir="./", model_filename=save_file_name)
        #
        # quantized_jit_model = self.load_torchscript_model(model_filepath=self.experiment_name + "_int8.pt", device='cpu')
        #
        # _, fp32_eval_accuracy = self.evaluate_model(model=original_model, test_loader=self.dataloader['test'], device='cpu', criterion=None)
        # _, int8_eval_accuracy = self.evaluate_model(model=model_int8, test_loader=self.dataloader['test'], device='cpu',
        #                                        criterion=None)
        # _, int8_jit_eval_accuracy = self.evaluate_model(model=quantized_jit_model, test_loader=self.dataloader['test'],
        #                                             device='cpu', criterion=None)
        # print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
        # print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
        # print("INT8 JIT evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
        #
        #
        # fp32_cpu_inference_latency = self.measure_inference_latency(model=original_model, device='cpu',
        #                                                        input_data=dummy_input, num_samples=100)
        # int8_cpu_inference_latency = self.measure_inference_latency(model=model_int8, device='cpu',
        #                                                        input_data=dummy_input, num_samples=100)
        # int8_jit_cpu_inference_latency = self.measure_inference_latency(model=quantized_jit_model, device='cpu',
        #                                                            input_data=dummy_input, num_samples=100)
        # fp32_gpu_inference_latency = self.measure_inference_latency(model=original_model, device='cuda',
        #                                                        input_data=dummy_input, num_samples=100)
        #
        # print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
        # print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
        # print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
        # print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

        return model_int8, model_fp32_trained

    # def compare_models(self, original_dense_model, pruned_fine_tuned_model, quantization=False,accuracies=None):

    def compare_models(self, model_list, model_tags=None):
        """
        Compares the performance of two PyTorch models: an original dense model and a pruned and fine-tuned model.
        Prints a table of metrics including latency, MACs, and model size for both models and their reduction ratios.

        Args:
        - original_dense_model: a PyTorch model object representing the original dense model
        - pruned_fine_tuned_model: a PyTorch model object representing the pruned and fine-tuned model

        Returns: None
        """

        table_data = {
            "latency": ["Latency (ms/sample)"],
            "accuracy": ["Accuracy (%)"],
            "params": ["Params (M)"],
            "size": ["Size (MiB)"],
            "mac": ["MAC (M)"],
            # "energy": ["Energy (Joules)"],
        }
        table = PrettyTable()
        table.field_names = ["", "Original Model", "Pruned Model", "Quantized Model"]

        accuracies, latency, params, model_size, macs = [], [], [], [], []
        skip = 3

        input_shape = list(next(iter(self.dataloader["test"]))[0].size())
        input_shape[0] = 1
        dummy_input = torch.randn(input_shape).to("cpu")
        file_name_list = ["original_model", "pruned_model", "quantized_model"]
        if not os.path.exists("weights/"):
            os.makedirs("weights/")
        for model, model_file_name in zip(model_list, file_name_list):
            # print(str(model))
            # # Parse through snn model and send to cpu
            if self.snn:
                if isinstance(model, nn.Sequential):
                    for layer_id in range(len(model)):
                        layer = model[layer_id]
                        if isinstance((layer), snntorch._neurons.leaky.Leaky):
                            layer.mem = layer.mem.to("cpu")
                else:
                    for module in model.modules():
                        if isinstance((layer), snntorch._neurons.leaky.Leaky):
                            layer.mem = layer.mem.to("cpu")

            table_data["accuracy"].append(
                round(self.evaluate(model=model, device="cpu"), 3)
            )
            table_data["latency"].append(
                round(
                    self.measure_latency(model=model.to("cpu"), dummy_input=dummy_input)
                    * 1000,
                    1,
                )
            )
            table_data["size"].append(self.get_model_size_weights(model))

            try:
                model_params = self.get_num_parameters(model, count_nonzero_only=True)
                if torch.is_tensor(model_params):
                    model_params = model_params.item()
                model_params = round(model_params / 1e6, 2)
                if model_params == 0:
                    table_data["params"].append("*")
                else:
                    table_data["params"].append(model_params)
            except RuntimeError as e:
                table_data["params"].append("*")
            if skip == 1:
                table_data["mac"].append("*")
                pass
            else:
                try:
                    mac = self.get_model_macs(model, dummy_input)
                    table_data["mac"].append(round(mac / 1e6))
                except AttributeError as e:
                    table_data["mac"].append("-")

            ########################################

            folder_file_name = "weights/" + model_file_name + "/"
            if not os.path.exists(folder_file_name):
                os.makedirs(folder_file_name)
            folder_file_name += model_file_name
            torch.save(model, folder_file_name + ".pt")
            torch.save(model.state_dict(), folder_file_name + "_weights.pt")
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(torch.jit.script(traced_model), folder_file_name + ".pt")

            ########################################
            # Save model with pt,.pth and jit
            skip -= 1

        for key, value in table_data.items():
            table.add_row(value)
        print(
            "\n \n============================== Comparison Table =============================="
        )
        print(table)

        # # accuracies = accuracies
        # if( accuracies == None):
        #     accuracies =[]
        #     accuracies.append(self.evaluate(model = original_dense_model))
        #     accuracies.append(self.evaluate(model = pruned_fine_tuned_model))

        # input_shape = list(next(iter(self.dataloader["test"]))[0].size())
        # input_shape[0] = 1
        # dummy_input = torch.randn(input_shape).to("cpu")
        # original_dense_model.to("cpu")
        # pruned_fine_tuned_model.to("cpu")

        # # Parse through snn model and send to cpu
        # if self.snn:
        #     for model in [original_dense_model, pruned_fine_tuned_model]:
        #         if isinstance(model, nn.Sequential):
        #             for layer_id in range(len(original_dense_model)):
        #                 layer = original_dense_model[layer_id]
        #                 if isinstance((layer), snntorch._neurons.leaky.Leaky):
        #                     layer.mem = layer.mem.to("cpu")
        #         else:
        #             for module in model.modules():
        #                 if isinstance((layer), snntorch._neurons.leaky.Leaky):
        #                     layer.mem = layer.mem.to("cpu")

        # original_dense_model.to("cpu")
        # original_latency = self.measure_latency(
        #     model=original_dense_model, dummy_input=dummy_input
        # )
        # pruned_latency = self.measure_latency(
        #     model=pruned_fine_tuned_model, dummy_input=dummy_input
        # )
        # table_struct = "Quantized" if (quantization==True) else "Pruned"
        # print("\n ................. Comparison Table  .................")
        # table_template = "{:<15} {:<15} {:<15} {:<15}"
        # print(table_template.format("", "Original", table_struct, "Reduction Ratio"))
        # print(
        #     table_template.format(
        #         "Latency (ms)",
        #         round(original_latency * 1000, 1),
        #         round(pruned_latency * 1000, 1),
        #         round(original_latency / pruned_latency, 1),
        #     )
        # )

        # # 2. measure the computation (MACs)
        # if(not quantization):
        #     original_macs = self.get_model_macs(original_dense_model, dummy_input)
        #     pruned_macs = self.get_model_macs(pruned_fine_tuned_model, dummy_input)
        #     table_template = "{:<15} {:<15} {:<15} {:<15}"
        #     print(
        #         table_template.format(
        #             "MACs (M)",
        #             round(original_macs / 1e6),
        #             round(pruned_macs / 1e6),
        #             round(original_macs / pruned_macs, 1),
        #         )
        #     )

        #     # 3. measure the model size (params)
        #     original_param = self.get_num_parameters(
        #         original_dense_model, count_nonzero_only=True
        #     ).item()
        #     pruned_param = self.get_num_parameters(
        #         pruned_fine_tuned_model, count_nonzero_only=True
        #     ).item()
        #     print(
        #         table_template.format(
        #             "Param (M)",
        #             round(original_param / 1e6, 2),
        #             round(pruned_param / 1e6, 2),
        #             round(original_param / pruned_param, 1),
        #         )
        #     )

        # # 4. Accuracies

        # print(
        #     table_template.format(
        #         "Accuracies (%)",
        #         round(accuracies[0], 3),
        #         round(accuracies[-1], 3),
        #         str(round(accuracies[-1] - accuracies[0], 3)),
        #     )
        # )

        # # put model back to cuda
        # # pruned_model = pruned_fine_tuned_model.to("cuda")
        # # model = original_dense_model.to("cuda")

    def CWP_Pruning(self):
        """
        Applies channel pruning to the model using the specified channel pruning ratio.
        Returns the pruned model.
        """
        sorted_model = self.apply_channel_sorting()
        # self.model = self.channel_prune(sorted_model, self.channel_pruning_ratio)
        self.sparsity_dict = [value for key, value in self.sparsity_dict.items()]
        self.model = self.channel_prune(sorted_model, self.sparsity_dict)

    def venum_CWP_Pruning(self, original_dense_model, sparsity_dict):
        """
        Applies channel pruning to the model using the specified channel pruning ratio.
        Returns the pruned model.
        """

        sparsity_dict = [value for key, value in sparsity_dict.items()]
        # #place original model below
        # sparsity_dict = sparsity_dict[:-2]
        self.model = self.channel_prune(original_dense_model, sparsity_dict)

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

        # iterate through conv layers
        for i_conv in range(len(all_convs) - 1):
            # each channel sorting index, we need to apply it to:
            # - the output dimension of the previous conv
            # - the previous BN layer
            # - the input dimension of the next conv (we compute importance here)
            prev_conv = all_convs[i_conv]
            prev_bn = all_bns[i_conv]
            next_conv = all_convs[i_conv + 1]
            # note that we always compute the importance according to input channels
            importance = self.get_input_channel_importance(next_conv.weight)
            # sorting from large to small
            sort_idx = torch.argsort(importance, descending=True)

            # apply to previous conv and its following bn
            prev_conv.weight.copy_(
                torch.index_select(prev_conv.weight.detach(), 0, sort_idx)
            )
            for tensor_name in ["weight", "bias", "running_mean", "running_var"]:
                tensor_to_apply = getattr(prev_bn, tensor_name)
                tensor_to_apply.copy_(
                    torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
                )

            # apply to the next conv input (hint: one line of code)
            ##################### YOUR CODE STARTS HERE #####################
            next_conv.weight.copy_(
                torch.index_select(next_conv.weight.detach(), 1, sort_idx)
            )

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
        if not isinstance(prune_ratio, list):
            prune_ratio = [prune_ratio] * (n_conv - 1)

        assert len(all_convs) == len(all_bns)

        for i_ratio, p_ratio in enumerate(prune_ratio):
            prev_conv = all_convs[i_ratio]
            prev_bn = all_bns[i_ratio]
            next_conv = all_convs[i_ratio + 1]
            original_channels = prev_conv.out_channels  # same as next_conv.in_channels
            if self.prune_mode != "venum_cwp":
                n_keep = self.get_num_channels_to_keep(original_channels, p_ratio)

                # prune the output of the previous conv and bn
                prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
                prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
                prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
                prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
                prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

                # prune the input of the next conv (hint: just one line of code)

                next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
            else:
                pick_list = self.venum_sorted_list[i_ratio]
                salient_indices = pick_list[int(original_channels * p_ratio) :]

                # prune the output of the previous conv and bn
                prev_conv.weight.set_(prev_conv.weight.detach()[salient_indices])
                prev_bn.weight.set_(prev_bn.weight.detach()[salient_indices])
                prev_bn.bias.set_(prev_bn.bias.detach()[salient_indices])
                prev_bn.running_mean.set_(
                    prev_bn.running_mean.detach()[salient_indices]
                )
                prev_bn.running_var.set_(prev_bn.running_var.detach()[salient_indices])

                # prune the input of the next conv (hint: just one line of code)
                ##################### YOUR CODE STARTS HERE #####################
                next_conv.weight.set_(next_conv.weight.detach()[:, salient_indices])

        return new_model
