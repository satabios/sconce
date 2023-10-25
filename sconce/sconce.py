
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

from snntorch import utils
from collections import namedtuple
from fast_pytorch_kmeans import KMeans
from torch.nn import parameter

import ipdb

from snntorch import functional as SF

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
        self.Codebook = namedtuple('Codebook', ['centroids', 'labels'])
        self.codebook = None
        self.channel_pruning_ratio = None
        self.snn = False
        self.accuracy_function = None

        self.bitwidth=4

        self.device = None


      ## Sparse GPT creds
        # self.layer = layer
        # self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        # if isinstance(self.layer, nn.Conv2d):
        #     W = W.flatten(1)
        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()
        # self.rows = W.shape[0]
        # self.columns = W.shape[1]
        # self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        # self.nsamples = 0

    def forward_pass_snn(self, data, mem_out_rec=None):
        spk_rec = []
        mem_rec = []
        utils.reset(self.model)  # resets hidden states for all LIF neurons in net

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out, mem_out = self.model(data[step])
            spk_rec.append(spk_out)
            if(mem_out_rec is not None):
                mem_rec.append(mem_out)
        if(mem_out_rec is not None):
            return torch.stack(spk_rec), torch.stack(mem_rec)
        else:
            return torch.stack(spk_rec)
    def train(self) -> None:

            torch.cuda.empty_cache()

            self.model.to(self.device)

            val_acc = 0
            running_loss = 0.0


            epochs = self.epochs if self.fine_tune==False  else self.num_finetune_epochs
            for epoch in range(epochs):

                self.model.train()

                validation_acc = 0

                for i, data in enumerate(tqdm(self.dataloader['train'], desc='train', leave=False)):
                    # Move the data from CPU to GPU
                    if (self.goal != 'autoencoder'):
                        inputs, targets = data
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    elif (self.goal == 'autoencoder'):
                        inputs, targets = data.to(self.device), data.to(self.device)

                    # Reset the gradients (from the last iteration)
                    self.optimizer.zero_grad()

                    # Forward inference
                    if (self.snn == True):
                        outputs = self.forward_pass_snn(inputs)
                        SF.accuracy_rate(outputs, targets) / 100

                    else:
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
                    torch.save( copy.deepcopy(self.model.state_dict()), self.experiment_name + '.pth')

    @torch.no_grad()
    def evaluate(self, verbose=False):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            local_acc = []
            for i, data in enumerate(tqdm(self.dataloader['test'], desc='test', leave=False)):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                if(self.snn):
                    outputs = self.forward_pass_snn(images, mem_out_rec=None)
                    correct += SF.accuracy_rate(outputs, labels) / 100
                else:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0) - 1
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            if (verbose):
                print('Test Accuracy: {} %'.format(acc))
            return acc

    ########## Model Profiling ##########
    def get_model_macs(self, model, inputs) -> int:
      return profile_macs(model, inputs)

    def get_sparsity(self, tensor: torch.Tensor) -> float:
      """
      calculate the sparsity of the given tensor
          sparsity = #zeros / #elements = 1 - #nonzeros / #elements
      """
      return 1 - float(tensor.count_nonzero()) / tensor.numel()

    def get_model_sparsity(self, model: nn.Module) -> float:
      """
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
      calculate the total number of parameters of model
      :param count_nonzero_only: only count nonzero weights
      """
      num_counted_elements = 0
      for param in model.parameters():
        if count_nonzero_only:
          num_counted_elements += param.count_nonzero()
        else:
          num_counted_elements += param.numel()
      return num_counted_elements

    def get_model_size(self, model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
      """
      calculate the model size in bits
      :param data_width: #bits per element
      :param count_nonzero_only: only count nonzero weights
      """
      return self.get_num_parameters(model, count_nonzero_only) * data_width

    @torch.no_grad()
    def measure_latency(self, model, dummy_input, n_warmup=20, n_test=100):
        model.to("cpu")
        model.eval()
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
    def GMP_apply(self):

        for name, param in self.model.named_parameters():
            if name in self.masks:
                param *= self.masks[name].to(self.device)

    # @staticmethod
    @torch.no_grad()
    def GMP_Pruning(self):
        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                self.masks[name] = self.fine_grained_prune(param, self.sparsity_dict[name])


    def compress(self, verbose=True) -> None:
        original_dense_model = self.model
        # self.train()

        dense_model_size = self.get_model_size(model = self.model,count_nonzero_only=True)
        print(f"\nDense_model_size model after sensitivity size={dense_model_size / MiB:.2f} MiB")
        dense_validation_acc = self.evaluate(verbose=False)
        print("Original Model Validation Accuracy:", dense_validation_acc, "%")
        self.dense_model_valid_acc = dense_validation_acc


        if(self.prune_mode == "GMP"):
          print("Granular-Magnitude Pruning")
          self.sensitivity_scan(dense_model_accuracy=dense_validation_acc, verbose=False)
          self.GMP_Pruning()  #FineGrained Pruning
          self.callbacks = [lambda: self.GMP_apply()]
          print(f"Sparsity for each Layer: {self.sparsity_dict}")
        elif(self.prune_mode == "CWP"):
          print("Channel-Wise Pruning")
          self.CWP_Pruning()  #Channelwise Pruning
        pruned_model_size = self.get_model_size(model = self.model, count_nonzero_only=True)
        # validation_acc = self.evaluate(verbose=False)
        print(f"Pruned model has size={pruned_model_size / MiB:.2f} MiB = {pruned_model_size / dense_model_size * 100:.2f}% of Original model size")
        #### Add Accuracy deviation

        self.fine_tune = True
        if (self.fine_tune):
          self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
          self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.num_finetune_epochs)
          self.train()
        fine_tuned_pruned_model_size = self.get_model_size(model = self.model , count_nonzero_only=True)
        validation_acc = self.evaluate(verbose=False)
        pruned_fine_tuned_model = self.model
        torch.save(copy.deepcopy(original_dense_model.state_dict()), self.experiment_name + '.pth')
        torch.save(copy.deepcopy(pruned_fine_tuned_model.state_dict()), self.experiment_name+'_pruned' + '.pth')

        self.compare_models(original_dense_model, pruned_fine_tuned_model)
        if (verbose):
            print(
                f"Fine-Tuned Sparse model has size={fine_tuned_pruned_model_size / MiB:.2f} MiB = {fine_tuned_pruned_model_size / dense_model_size * 100:.2f}% of Original model size")
            print("Fine-Tuned Pruned Model Validation Accuracy:", validation_acc)


#make this change to sync the files to beast; did that affect it?

    def k_means_quantize(self, fp32_tensor: torch.Tensor, bitwidth=4,codebook=None):

        """
        quantize tensor using k-means clustering
        :param fp32_tensor:
        :param bitwidth: [int] quantization bit width, default=4
        :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
        :return:
            [Codebook = (centroids, labels)]
                centroids: [torch.(cuda.)FloatTensor] the cluster centroids
                labels: [torch.(cuda.)LongTensor] cluster label tensor
        """

        if codebook is None:
          # get number of clusters based on the quantization precision
          # hint: one line of code
          n_clusters = 1 << self.bitwidth

          # use k-means to get the quantization centroids
          kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
          labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
          centroids = kmeans.centroids.to(torch.float).view(-1)
          self.codebook = self.Codebook(centroids, labels)

        # decode the codebook into k-means quantized tensor for inference
        # hint: one line of code
        # ipdb.set_trace()
        quantized_tensor = self.codebook.centroids[self.codebook.labels]

        fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
        return codebook



    @torch.no_grad()
    def kmeansquantize(self):
      codebook = dict()
      if isinstance(self.bitwidth, dict):
        for name, param in self.model.named_parameters():
          if name in bitwidth:
            codebook[name] = self.k_means_quantize(param, bitwidth=self.bitwidth[name])
      else:
        for name, param in self.model.named_parameters():
          if param.dim() > 1:
            codebook[name] = self.k_means_quantize(param, bitwidth=self.bitwidth)
      return codebook



      @torch.no_grad()
      def apply_quantization(self,   update_centroids):
        self.codebook = self.quantize()
        for name, param in self.model.named_parameters():
          if name in self.codebook:
            if update_centroids:
              update_codebook(param, codebook=self.codebook[name])
            self.codebook[name] = k_means_quantize(
              param, codebook=self.codebook[name])


    def compare_models(self, original_dense_model, pruned_fine_tuned_model):
      input_shape = list(next(iter(self.dataloader['test']))[0].size())
      input_shape[0] = 1
      dummy_input = torch.randn(input_shape).to('cpu')
      pruned_model = pruned_fine_tuned_model.to('cpu')
      model = original_dense_model.to('cpu')

      pruned_latency = self.measure_latency(model = pruned_model, dummy_input=dummy_input)
      original_latency = self.measure_latency(model = model, dummy_input=dummy_input)
      print("/n")
      table_template = "{:<15} {:<15} {:<15} {:<15}"
      print(table_template.format('', 'Original', 'Pruned', 'Reduction Ratio'))
      print(table_template.format('Latency (ms)',
                                  round(original_latency * 1000, 1),
                                  round(pruned_latency * 1000, 1),
                                  round(original_latency / pruned_latency, 1)))

      # 2. measure the computation (MACs)
      original_macs = self.get_model_macs(model, dummy_input)
      pruned_macs = self.get_model_macs(pruned_model, dummy_input)
      table_template = "{:<15} {:<15} {:<15} {:<15}"
      print(table_template.format('MACs (M)',
                                  round(original_macs / 1e6),
                                  round(pruned_macs / 1e6),
                                  round(original_macs / pruned_macs, 1)))

      # 3. measure the model size (params)
      original_param = self.get_num_parameters(model)
      pruned_param = self.get_num_parameters(pruned_model)
      print(table_template.format('Param (M)',
                                  round(original_param / 1e6, 2),
                                  round(pruned_param / 1e6, 2),
                                  round(original_param / pruned_param, 1)))

      # put model back to cuda
      pruned_model = pruned_model.to('cuda')
      model = model.to('cuda')



    ### Channel Pruning ###
    # function to sort the channels from important to non-important

    def CWP_Pruning(self):
      sorted_model = self.apply_channel_sorting()
      self.model = self.channel_prune(sorted_model, self.channel_pruning_ratio)


    def get_input_channel_importance(self, weight):
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
      model = copy.deepcopy(self.model)  # do not modify the original model
      # fetch all the conv and bn layers from the backbone

      all_convs = []
      all_bns = []

      #Universal Layer Seeking by Parsing
      def find_instance(obj, object_of_importance ):
        if isinstance(obj, object_of_importance):
          if(object_of_importance == nn.Conv2d):
            all_convs.append(obj)
          elif(object_of_importance == nn.BatchNorm2d):
            all_bns.append(obj)
          return None
        elif isinstance(obj, list):
          for internal_obj in obj:
            find_instance(internal_obj, object_of_importance)
        elif (hasattr(obj, '__class__')):
          for internal_obj in obj.children():
            find_instance(internal_obj, object_of_importance)
        elif isinstance(obj, OrderedDict):
          for key, value in obj.items():
            find_instance(value, object_of_importance)

      find_instance(obj = model, object_of_importance=nn.Conv2d)
      find_instance(obj = model, object_of_importance=nn.BatchNorm2d)

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
        prev_conv.weight.copy_(torch.index_select(
          prev_conv.weight.detach(), 0, sort_idx))
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
          tensor_to_apply = getattr(prev_bn, tensor_name)
          tensor_to_apply.copy_(
            torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
          )

        # apply to the next conv input (hint: one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.copy_(
          torch.index_select(next_conv.weight.detach(), 1, sort_idx))
        ##################### YOUR CODE ENDS HERE #####################

      return model

    def get_num_channels_to_keep(self, channels: int, prune_ratio: float) -> int:
        """A function to calculate the number of layers to PRESERVE after pruning
        Note that preserve_rate = 1. - prune_ratio
        """
        ##################### YOUR CODE STARTS HERE #####################
        return int(round(channels * (1. - prune_ratio)))
        ##################### YOUR CODE ENDS HERE #####################

    def get_num_channels_to_keep(self, channels: int, prune_ratio: float) -> int:
      """A function to calculate the number of layers to PRESERVE after pruning
      Note that preserve_rate = 1. - prune_ratio
      """
      ##################### YOUR CODE STARTS HERE #####################
      return int(round(channels * (1. - prune_ratio)))
      ##################### YOUR CODE ENDS HERE #####################

    @torch.no_grad()
    def channel_prune(self,model: nn.Module,
                      prune_ratio: Union[List, float]) -> nn.Module:
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
          if (object_of_importance == nn.Conv2d):
            all_convs.append(obj)
          elif (object_of_importance == nn.BatchNorm2d):
            all_bns.append(obj)
          return None
        elif isinstance(obj, list):
          for internal_obj in obj:
            find_instance(internal_obj, object_of_importance)
        elif (hasattr(obj, '__class__')):
          for internal_obj in obj.children():
            find_instance(internal_obj, object_of_importance)
        elif isinstance(obj, OrderedDict):
          for key, value in obj.items():
            find_instance(value, object_of_importance)

      # we prune the convs in the backbone with a uniform ratio
      new_model = copy.deepcopy(model)  # prevent overwrite
      find_instance(obj= new_model, object_of_importance=nn.Conv2d)
      find_instance(obj= new_model, object_of_importance=nn.BatchNorm2d)
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





