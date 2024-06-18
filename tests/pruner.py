import torch
import numpy as np
from tqdm import tqdm
import copy
import torch.nn as nn
from collections import OrderedDict, defaultdict
from typing import Union, List

class prune:
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
        param_names = [i[0] for i in named_all_weights]
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
                elif self.prune_mode == "GMP":
                    sparse_list = np.zeros(len(named_all_weights))
                    sparse_list[i_layer] = sparsity
                    local_sparsity_dict = dict(zip(param_names, sparse_list))
                    self.GMP_Pruning(
                        prune_dict=local_sparsity_dict
                    )  # FineGrained Pruning
                    self.callbacks = [lambda: self.GMP_apply()]
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
                    self.model = copy.deepcopy(original_model)
                    if abs(acc) <= self.degradation_value:
                        self.sparsity_dict[name] = sparsity
                        self.model = copy.deepcopy(original_model)
                        break
                    elif sparsity == scan_start:
                        accuracy = np.asarray(accuracy)

                        if np.max(accuracy) > -0.75:  # Allowed Degradation
                            acc_x = np.where(accuracy == np.max(accuracy))[0][0]
                            best_possible_sparsity = sparsities[acc_x]

                        else:
                            best_possible_sparsity = 0
                        self.sparsity_dict[name] = best_possible_sparsity
                        self.model = copy.deepcopy(original_model)
                    else:
                        accuracy.append(acc)
                        hit_flag = False
                    self.model = copy.deepcopy(original_model)

    def fine_grained_prune(self, tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
        """
        Magnitude-based pruning for single tensor

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
    def GMP_Pruning(self, model=None, prune_dict=None):
        """
        Applies Group-wise Magnitude Pruning (GMP) to the model's convolutional and fully-connected weights.
        The pruning is performed based on the sparsity levels specified in the `sparsity_dict` attribute.
        The pruned weights are stored in the `masks` attribute.
        """
        if prune_dict != None:
            sparse_dict = prune_dict
        else:
            sparse_dict = self.sparsity_dict

        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                self.masks[name] = self.fine_grained_prune(param, sparse_dict[name])

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

            importance = torch.norm(channel_weight)

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

            next_conv.weight.copy_(
                torch.index_select(next_conv.weight.detach(), 1, sort_idx)
            )

        return model

    def get_num_channels_to_keep(self, channels: int, prune_ratio: float) -> int:
        """A function to calculate the number of layers to PRESERVE after pruning
        Note that preserve_rate = 1. - prune_ratio
        """

        return int(round(channels * (1.0 - prune_ratio)))

    def get_num_channels_to_keep(self, channels: int, prune_ratio: float) -> int:
        """A function to calculate the number of layers to PRESERVE after pruning
        Note that preserve_rate = 1. - prune_ratio
        """

        return int(round(channels * (1.0 - prune_ratio)))

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

        next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])

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
                if prev_conv.bias is not None:
                    prev_conv.bias = nn.Parameter(prev_conv.bias.detach()[:n_keep])
                prev_conv.out_channels = n_keep

                prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
                prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
                prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
                prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])
                prev_bn.num_features = n_keep

                # prune the input of the next conv (hint: just one line of code)

                next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
                next_conv.in_channels = n_keep
            else:
                pick_list = self.venum_sorted_list[i_ratio]
                salient_indices = pick_list[int(original_channels * p_ratio):]

                # prune the output of the previous conv and bn
                prev_conv.weight.set_(prev_conv.weight.detach()[salient_indices])
                prev_bn.weight.set_(prev_bn.weight.detach()[salient_indices])
                prev_bn.bias.set_(prev_bn.bias.detach()[salient_indices])
                prev_bn.running_mean.set_(
                    prev_bn.running_mean.detach()[salient_indices]
                )
                prev_bn.running_var.set_(prev_bn.running_var.detach()[salient_indices])

                # prune the input of the next conv (hint: just one line of code)

                next_conv.weight.set_(next_conv.weight.detach()[:, salient_indices])

        return new_model

    def CWP_Pruning(self):
        """
        Applies channel pruning to the model using the specified channel pruning ratio.
        Returns the pruned model.
        """
        sorted_model = self.apply_channel_sorting()
        # self.model = self.channel_prune(sorted_model, self.channel_pruning_ratio)
        self.sparsity_dict = [value for key, value in self.sparsity_dict.items()]
        self.model = self.channel_prune(sorted_model, self.sparsity_dict)