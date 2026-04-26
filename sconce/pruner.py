"""
sconce/pruner.py
Unified pruner: CNN magnitude pruning (GMP), channel-wise pruning (CWP),
and Transformer structural pruning — all in one place.

Shared low-level utilities live in utils.py and are imported below.
"""
from __future__ import annotations

import copy
import queue
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import (
    PRUNABLE_MODULES,
    _build_gpu_worker_pool,
    _collect_conv_bn,
    _cwp_prune_module,
    _get_example_inputs,
    _measure_model_vram_gb,
    _nvidia_smi_free_gb,
    _structured_zero_prune,
)


class prune:

    # @torch.no_grad()
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

        # Create a deep copy of the model first
        original_model = copy.deepcopy(self.model)

        # Generate named_all_weights from original_model instead of self.model
        named_all_weights = [
            (name, param)
            for (name, param) in original_model.named_parameters()
            if param.dim() > 1
        ]
        param_names = [i[0] for i in named_all_weights]
        original_prune_mode = self.prune_mode

        # Initialize variables for CWP pruning
        if self.prune_mode == "CWP":
            model_device = next(original_model.parameters()).device
            # Use original_model to define named_all_weights for CWP
            named_all_weights = [
                (name, module)
                for name, module in original_model.named_modules()
                if isinstance(module, PRUNABLE_MODULES)
            ]
            example_inputs = _get_example_inputs(self.dataloader['test'], model_device)

        layer_iter = tqdm(named_all_weights, desc="layer", leave=False)

        for i_layer, (name, param_or_module) in enumerate(layer_iter):
            accuracy = []
            desc = f"scanning {i_layer}/{len(named_all_weights)} - {name}" if verbose else None
            picker = tqdm(sparsities, desc=desc) if verbose else sparsities
            hit_flag = False

            for sparsity in picker:
                # Reset model to original state at each sparsity step
                self.model = copy.deepcopy(original_model)

                # Retrieve the current parameter/module from the fresh model copy
                if self.prune_mode == "CWP":
                    current_module = dict(self.model.named_modules())[name]
                    _cwp_prune_module(self.model, current_module, sparsity, example_inputs)
                    hit_flag = True

                elif self.prune_mode == "GMP":
                    # For GMP, sparsity is applied directly to the parameter
                    current_param = dict(self.model.named_parameters())[name]
                    sparse_list = np.zeros(len(named_all_weights))
                    sparse_list[i_layer] = sparsity
                    local_sparsity_dict = dict(zip(param_names, sparse_list))
                    self.GMP_Pruning(prune_dict=local_sparsity_dict)
                    self.callbacks = [lambda: self.GMP_apply()]
                    hit_flag = True

                if hit_flag:
                    # Evaluate the pruned model
                    acc = self.evaluate(Tqdm=False) - dense_model_accuracy
                    if abs(acc) <= (self.degradation_value)/3:
                        self.sparsity_dict[name] = sparsity
                        break
                    elif sparsity == sparsities[-1]:  # Last sparsity step
                        if accuracy:  # Handle edge case where no sparsity met the condition
                            best_sparsity = sparsities[np.argmax(accuracy)]
                            self.sparsity_dict[name] = best_sparsity if np.max(accuracy) > -0.60 else 0.0
                        else:
                            self.sparsity_dict[name] = 0.0
                    else:
                        accuracy.append(acc)

        # Restore original model and prune mode
        self.model = original_model
        self.prune_mode = original_prune_mode
        return self.sparsity_dict

    def sensitivity_scan_parallel(
            self,
            dense_model_accuracy,
            scan_step=0.05,
            scan_start=0.1,
            scan_end=1.0,
            verbose=True,
            safety_net_gb=2.0,
            activation_multiplier=1.5,
    ):
        """GPU-parallel sensitivity scan.

        Same interface as sensitivity_scan() but dispatches all
        (layer, sparsity) jobs across all available GPUs concurrently.
        Falls back to serial scan if no CUDA GPUs are found.

        Extra args:
            safety_net_gb: VRAM headroom reserved per GPU (default 2 GB).
            activation_multiplier: factor applied to measured model VRAM to
                account for activation buffers during eval (default 1.5).
        """
        if self.snn:
            # SNN forward pass not yet supported in parallel path.
            return self.sensitivity_scan(dense_model_accuracy, scan_step, scan_start, scan_end, verbose)

        if not torch.cuda.is_available():
            return self.sensitivity_scan(dense_model_accuracy, scan_step, scan_start, scan_end, verbose)

        self.sparsity_dict = {}
        sparsities = list(np.flip(np.arange(start=scan_start, stop=scan_end, step=scan_step)))
        original_model = copy.deepcopy(self.model)
        original_prune_mode = self.prune_mode

        # ── Build layer list (mirrors serial version) ──────────────────────
        if self.prune_mode == "CWP":
            named_all_weights = [
                (name, module)
                for name, module in original_model.named_modules()
                if isinstance(module, PRUNABLE_MODULES)
            ]
            example_inputs_cpu = _get_example_inputs(self.dataloader['test'])
        else:
            named_all_weights = [
                (name, param)
                for name, param in original_model.named_parameters()
                if param.dim() > 1
            ]
            example_inputs_cpu = None

        layer_names = [n for n, _ in named_all_weights]

        # ── GPU discovery + worker allocation ──────────────────────────────
        gpu_list = _nvidia_smi_free_gb()
        if not gpu_list:
            return self.sensitivity_scan(dense_model_accuracy, scan_step, scan_start, scan_end, verbose)

        gpu_queue, total_slots, _ = _build_gpu_worker_pool(
            gpu_list, original_model, safety_net_gb, activation_multiplier,
            verbose, "parallel sensitivity scan",
        )
        total_jobs = len(layer_names) * len(sparsities)
        if verbose:
            print(f"  Total slots: {total_slots}  |  jobs: {total_jobs}  ({len(layer_names)} layers × {len(sparsities)} sparsities)")

        # ── Worker function ────────────────────────────────────────────────
        results: list[tuple[str, float, float]] = []  # (layer_name, sparsity, acc_drop)
        results_lock = threading.Lock()
        print_lock = threading.Lock()

        def _worker(layer_name: str, sparsity: float) -> None:
            gpu_id = gpu_queue.get()
            device = torch.device(f"cuda:{gpu_id}")
            try:
                model_copy = copy.deepcopy(original_model).to(device)

                if original_prune_mode == "CWP":
                    # Move example_inputs to worker device — handles both Tensor and dict
                    if isinstance(example_inputs_cpu, dict):
                        ex_in = {k: v.to(device) for k, v in example_inputs_cpu.items()}
                    else:
                        ex_in = example_inputs_cpu.to(device)
                    current_module = dict(model_copy.named_modules())[layer_name]
                    _cwp_prune_module(model_copy, current_module, sparsity, ex_in)

                elif original_prune_mode == "GMP":
                    for pname, param in model_copy.named_parameters():
                        if pname == layer_name and param.dim() > 1:
                            self.fine_grained_prune(param, sparsity)
                            break

                _, acc_tensor = self.evaluate_model(model_copy, self.dataloader['test'], device)
                acc_pct = float(acc_tensor) * 100.0
                drop = acc_pct - dense_model_accuracy

                with results_lock:
                    results.append((layer_name, sparsity, drop))
                if verbose:
                    with print_lock:
                        print(f"  [GPU {gpu_id}] {layer_name}  sp={sparsity:.3f}  drop={drop:+.2f}%")

            except Exception as exc:
                with print_lock:
                    print(f"  [GPU {gpu_id}] {layer_name}  sp={sparsity:.3f}  ERROR: {exc}")
            finally:
                try:
                    del model_copy
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gpu_queue.put(gpu_id)

        # ── Dispatch ───────────────────────────────────────────────────────
        all_jobs = [(name, sp) for name in layer_names for sp in sparsities]
        with ThreadPoolExecutor(max_workers=total_slots) as pool:
            futures = [pool.submit(_worker, name, sp) for name, sp in all_jobs]
            for f in as_completed(futures):
                exc = f.exception()
                if exc is not None and verbose:
                    with print_lock:
                        print(f"  [worker exception] {exc}")

        # ── Post-process: build sparsity_dict ─────────────────────────────
        layer_results: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (lname, sp, drop) in results:
            layer_results[lname].append((sp, drop))

        for layer_name in layer_names:
            data = sorted(layer_results.get(layer_name, []), key=lambda x: x[0], reverse=True)
            best_sp = 0.0
            best_partial: tuple[float, float] | None = None
            for sp, drop in data:
                if abs(drop) <= self.degradation_value / 3:
                    best_sp = sp
                    break
                if best_partial is None or drop > best_partial[1]:
                    best_partial = (sp, drop)
            if best_sp == 0.0 and best_partial is not None and best_partial[1] > -0.60:
                best_sp = best_partial[0]
            self.sparsity_dict[layer_name] = best_sp

        self.model = original_model
        self.prune_mode = original_prune_mode
        return self.sparsity_dict

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

        num_zeros = round(num_elements * sparsity)
        importance = tensor.abs()
        threshold = importance.view(-1).kthvalue(num_zeros).values
        mask = torch.gt(importance, threshold)

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
        """Return L2-norm importance per input channel of *weight* [out, in, ...]."""
        return weight.detach().view(weight.shape[0], weight.shape[1], -1).norm(dim=(0, 2))

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

        all_convs, all_bns = _collect_conv_bn(model)

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

        # we prune the convs in the backbone with a uniform ratio
        new_model = copy.deepcopy(model)  # prevent overwrite
        all_convs, all_bns = _collect_conv_bn(new_model)
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

        # we prune the convs in the backbone with a uniform ratio
        new_model = copy.deepcopy(model)  # prevent overwrite
        all_convs, all_bns = _collect_conv_bn(new_model)
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
        model_device = next(self.model.parameters()).device
        example_inputs = _get_example_inputs(self.dataloader['test'], model_device)
        prune_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, PRUNABLE_MODULES)
        ]
        ratio_dict = {module: sparsity for (name, module), (name, sparsity) in zip(prune_modules, self.sparsity_dict.items())}

        try:
            import torch_pruning as tp
            pruner = tp.pruner.MetaPruner(
                self.model,
                example_inputs,
                importance=tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
                pruning_ratio=0,
                pruning_ratio_dict=ratio_dict,
                round_to=8,
            )
            pruner.step()
        except ImportError:
            for module, sparsity in ratio_dict.items():
                _structured_zero_prune(module, sparsity)


# ===========================================================================
# Transformer Structural Pruning
# (formerly transformer_pruner.py — merged here for unified pruning API)
# ===========================================================================
# ---------------------------------------------------------------------------
# 1. Name registries
# ---------------------------------------------------------------------------

FUSED_QKV_NAMES   = ("qkv", "c_attn", "in_proj", "qkv_proj", "Wqkv", "query_key_value")
Q_NAMES           = ("q_proj", "q", "query", "wq", "q_lin")
K_NAMES           = ("k_proj", "k", "key",   "wk", "k_lin")
V_NAMES           = ("v_proj", "v", "value", "wv", "v_lin")
OUT_PROJ_NAMES    = ("proj", "o_proj", "out_proj", "c_proj", "dense", "out")

FFN_UP_NAMES      = ("fc1", "dense", "c_fc", "wi", "up_proj", "w1")
FFN_DOWN_NAMES    = ("fc2", "dense_4h_to_h", "c_proj", "wo", "down_proj", "w2")
SWIGLU_GATE_NAMES = ("gate_proj", "fc1_g", "w1", "wi_0")
SWIGLU_UP_NAMES   = ("up_proj",   "fc1_x", "w3", "wi_1")
SWIGLU_DOWN_NAMES = ("down_proj", "fc2",   "w2", "wo")

# Block containers — checked in order; first nn.ModuleList match wins
BLOCK_PATH_SEQUENCES: List[List[str]] = [
    ["blocks"],              # timm ViT / EVA02
    ["layers"],              # Qwen2 / LLaMA (plain HF)
    ["model", "layers"],     # HF LLaMA: model.model.layers
    ["transformer", "h"],    # GPT-2
    ["encoder", "layer"],    # BERT / RoBERTa
    ["encoder", "layers"],   # InternViT / other HF vision encoders
    ["decoder", "layers"],   # T5 decoder
]

ATTN_NAMES = ("attn", "self_attn", "attention", "self_attention", "mha")
MLP_NAMES  = ("mlp", "ffn", "feed_forward", "intermediate", "pwff")

# Candidate divisors tried when inferring num_heads from weight shapes
_HEAD_DIVISORS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 64)
# Reasonable head-dim range for plausibility check
_MIN_HEAD_DIM, _MAX_HEAD_DIM = 8, 256

NUM_HEADS_ATTRS  = ("num_heads", "num_attention_heads", "n_heads", "n_head", "nhead")
HEAD_DIM_ATTRS   = ("head_dim", "attention_head_size", "d_head", "head_size")
NUM_KV_HEADS_ATTRS = ("num_key_value_heads", "num_kv_heads", "kv_heads", "num_query_groups")


# ---------------------------------------------------------------------------
# 2. Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AttentionSpec:
    """Describes the attention projection structure of one transformer block."""

    # Live module references — exactly one of fused_qkv or (q, k, v) is not None
    fused_qkv: Optional[nn.Linear]
    q: Optional[nn.Linear]
    k: Optional[nn.Linear]
    v: Optional[nn.Linear]
    out_proj: nn.Linear

    # Attribute names used to write back rebuilt modules via setattr
    fused_qkv_attr: Optional[str]
    q_attr: Optional[str]
    k_attr: Optional[str]
    v_attr: Optional[str]
    out_proj_attr: str

    # Dimension info
    num_heads: int
    head_dim: int
    embed_dim: int

    # GQA: num_kv_heads == num_heads for standard MHA
    num_kv_heads: int
    gqa_ratio: int      # = num_heads // num_kv_heads

    # Reference to the parent attention sub-module (e.g. block.attn)
    attn_module: nn.Module
    attn_attr: str       # attribute name used to reach attn_module from its block

    @property
    def is_fused(self) -> bool:
        return self.fused_qkv is not None

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_heads


@dataclass
class FFNSpec:
    """Describes the FFN structure of one transformer block."""

    ffn_type: str   # "standard" | "swiglu"

    # Standard two-layer FFN
    fc_up:        Optional[nn.Linear]
    fc_up_attr:   Optional[str]
    fc_down:      Optional[nn.Linear]
    fc_down_attr: Optional[str]

    # SwiGLU three-branch FFN
    gate:      Optional[nn.Linear]
    gate_attr: Optional[str]
    up:        Optional[nn.Linear]
    up_attr:   Optional[str]
    down:      Optional[nn.Linear]
    down_attr: Optional[str]

    # Optional intermediate LayerNorm (EVA02 SwiGLU style)
    norm:      Optional[nn.Module]
    norm_attr: Optional[str]

    intermediate_dim: int
    mlp_module: nn.Module
    mlp_attr: str


@dataclass
class TransformerLayerSpec:
    """Combines attention and FFN specs for a single transformer block."""

    block_idx: int
    block: nn.Module
    attn: AttentionSpec
    ffn: FFNSpec


# ---------------------------------------------------------------------------
# 3. Detection helpers
# ---------------------------------------------------------------------------

def _find_block_container(model: nn.Module) -> Optional[nn.ModuleList]:
    """Walk BLOCK_PATH_SEQUENCES; return first matching nn.ModuleList/Sequential."""
    for path in BLOCK_PATH_SEQUENCES:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None and isinstance(node, (nn.ModuleList, nn.Sequential)):
            return node
    return None


def _find_linear_by_names(
    module: nn.Module, names: Tuple[str, ...]
) -> Tuple[Optional[nn.Linear], Optional[str]]:
    """Try each name in order; return (Linear, name) or (None, None)."""
    for name in names:
        m = getattr(module, name, None)
        if isinstance(m, nn.Linear):
            return m, name
    return None, None


def _find_submodule_by_names(
    module: nn.Module, names: Tuple[str, ...]
) -> Tuple[Optional[nn.Module], Optional[str]]:
    """Try each name in order; return first nn.Module match."""
    for name in names:
        m = getattr(module, name, None)
        if isinstance(m, nn.Module):
            return m, name
    return None, None


def _infer_num_heads(
    attn_module: nn.Module,
    out_dim: int,        # out_features of q_proj (separate) or qkv.out_features//3 (fused)
    embed_dim: int,
    model: nn.Module,
) -> int:
    """Try attribute lookup, then model config, then weight-shape divisor search."""
    # 1. Direct attribute
    for attr in NUM_HEADS_ATTRS:
        v = getattr(attn_module, attr, None)
        if isinstance(v, int) and v > 0:
            return v

    # 2. Model-level config (HF models carry model.config)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in NUM_HEADS_ATTRS:
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v

    # 3. head_dim attribute → num_heads = out_dim // head_dim
    for attr in HEAD_DIM_ATTRS:
        hd = getattr(attn_module, attr, None)
        if isinstance(hd, int) and hd > 0 and out_dim % hd == 0:
            return out_dim // hd

    # 4. Divisor search: find plausible (num_heads, head_dim) pair
    for nh in _HEAD_DIVISORS:
        if out_dim % nh == 0:
            hd = out_dim // nh
            if _MIN_HEAD_DIM <= hd <= _MAX_HEAD_DIM:
                # Extra plausibility: head_dim should divide embed_dim too
                if embed_dim % hd == 0:
                    return nh

    # Fallback: treat as single head
    return 1


def _infer_head_dim(attn_module: nn.Module, num_heads: int, out_dim: int) -> int:
    """head_dim = out_dim // num_heads, or from known attribute."""
    for attr in HEAD_DIM_ATTRS:
        v = getattr(attn_module, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    if num_heads > 0 and out_dim % num_heads == 0:
        return out_dim // num_heads
    return out_dim  # fallback: 1 head


def _infer_num_kv_heads(
    attn_module: nn.Module,
    num_heads: int,
    head_dim: int,
    k_proj: Optional[nn.Linear],
    model: nn.Module,
) -> int:
    """Return num_kv_heads (== num_heads for standard MHA, < num_heads for GQA)."""
    # 1. Direct attribute
    for attr in NUM_KV_HEADS_ATTRS:
        v = getattr(attn_module, attr, None)
        if isinstance(v, int) and v > 0:
            return v

    # 2. Model-level config
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in NUM_KV_HEADS_ATTRS:
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v

    # 3. Infer from k_proj weight shape (separate path only)
    if k_proj is not None and head_dim > 0 and k_proj.out_features % head_dim == 0:
        nkv = k_proj.out_features // head_dim
        if nkv != num_heads:
            return nkv

    return num_heads   # standard MHA fallback


def _build_attn_spec(
    block: nn.Module,
    attn_module: nn.Module,
    attn_attr: str,
    model: nn.Module,
) -> Optional[AttentionSpec]:
    """
    Detect fused vs separate QKV, output proj, head counts.
    Returns None if the module is unrecognised.
    """
    out_proj, out_proj_attr = _find_linear_by_names(attn_module, OUT_PROJ_NAMES)
    if out_proj is None:
        return None

    embed_dim = out_proj.out_features

    # ── Try fused QKV first ──────────────────────────────────────────────
    fused_qkv, fused_qkv_attr = _find_linear_by_names(attn_module, FUSED_QKV_NAMES)
    if fused_qkv is not None:
        # out_features = 3 * num_heads * head_dim
        out_dim = fused_qkv.out_features // 3
        num_heads = _infer_num_heads(attn_module, out_dim, embed_dim, model)
        head_dim  = _infer_head_dim(attn_module, num_heads, out_dim)
        # GQA not typical for fused QKV, but handle gracefully
        num_kv_heads = _infer_num_kv_heads(attn_module, num_heads, head_dim, None, model)
        gqa_ratio = max(1, num_heads // num_kv_heads)
        return AttentionSpec(
            fused_qkv=fused_qkv, q=None, k=None, v=None, out_proj=out_proj,
            fused_qkv_attr=fused_qkv_attr, q_attr=None, k_attr=None, v_attr=None,
            out_proj_attr=out_proj_attr,
            num_heads=num_heads, head_dim=head_dim, embed_dim=embed_dim,
            num_kv_heads=num_kv_heads, gqa_ratio=gqa_ratio,
            attn_module=attn_module, attn_attr=attn_attr,
        )

    # ── Try separate Q/K/V ───────────────────────────────────────────────
    q, q_attr = _find_linear_by_names(attn_module, Q_NAMES)
    k, k_attr = _find_linear_by_names(attn_module, K_NAMES)
    v, v_attr = _find_linear_by_names(attn_module, V_NAMES)

    if q is None or v is None:
        return None   # Can't identify projections

    out_dim = q.out_features
    num_heads = _infer_num_heads(attn_module, out_dim, embed_dim, model)
    head_dim  = _infer_head_dim(attn_module, num_heads, out_dim)
    num_kv_heads = _infer_num_kv_heads(attn_module, num_heads, head_dim, k, model)
    gqa_ratio = max(1, num_heads // num_kv_heads)

    return AttentionSpec(
        fused_qkv=None, q=q, k=k, v=v, out_proj=out_proj,
        fused_qkv_attr=None, q_attr=q_attr, k_attr=k_attr, v_attr=v_attr,
        out_proj_attr=out_proj_attr,
        num_heads=num_heads, head_dim=head_dim, embed_dim=embed_dim,
        num_kv_heads=num_kv_heads, gqa_ratio=gqa_ratio,
        attn_module=attn_module, attn_attr=attn_attr,
    )


def _build_ffn_spec(
    mlp_module: nn.Module, mlp_attr: str
) -> Optional[FFNSpec]:
    """Detect standard vs SwiGLU FFN. Returns None if unrecognised."""
    # ── SwiGLU detection: needs distinct gate AND up ────────────────────
    gate, gate_attr = _find_linear_by_names(mlp_module, SWIGLU_GATE_NAMES)
    up,   up_attr   = _find_linear_by_names(mlp_module, SWIGLU_UP_NAMES)
    down, down_attr = _find_linear_by_names(mlp_module, SWIGLU_DOWN_NAMES)

    is_swiglu = (gate is not None and up is not None and down is not None
                 and gate is not up)   # guard against same Linear resolved twice

    if is_swiglu:
        intermediate_dim = gate.out_features
        # Optional intermediate LayerNorm (EVA02 style)
        norm, norm_attr = None, None
        for n in ("norm", "ln", "layer_norm", "mid_norm"):
            cand = getattr(mlp_module, n, None)
            if isinstance(cand, nn.LayerNorm) and cand.normalized_shape[0] == intermediate_dim:
                norm, norm_attr = cand, n
                break
        return FFNSpec(
            ffn_type="swiglu",
            fc_up=None, fc_up_attr=None, fc_down=None, fc_down_attr=None,
            gate=gate, gate_attr=gate_attr,
            up=up,   up_attr=up_attr,
            down=down, down_attr=down_attr,
            norm=norm, norm_attr=norm_attr,
            intermediate_dim=intermediate_dim,
            mlp_module=mlp_module, mlp_attr=mlp_attr,
        )

    # ── Standard 2-layer FFN ────────────────────────────────────────────
    fc_up,   fc_up_attr   = _find_linear_by_names(mlp_module, FFN_UP_NAMES)
    fc_down, fc_down_attr = _find_linear_by_names(mlp_module, FFN_DOWN_NAMES)

    if fc_up is None or fc_down is None or fc_up is fc_down:
        return None

    return FFNSpec(
        ffn_type="standard",
        fc_up=fc_up, fc_up_attr=fc_up_attr,
        fc_down=fc_down, fc_down_attr=fc_down_attr,
        gate=None, gate_attr=None,
        up=None,   up_attr=None,
        down=None, down_attr=None,
        norm=None, norm_attr=None,
        intermediate_dim=fc_up.out_features,
        mlp_module=mlp_module, mlp_attr=mlp_attr,
    )


# ---------------------------------------------------------------------------
# 4. Public: layer discovery
# ---------------------------------------------------------------------------

def find_transformer_layers(model: nn.Module) -> List[TransformerLayerSpec]:
    """
    Walk *model* and return one ``TransformerLayerSpec`` per block.

    Works for any architecture whose blocks are reachable via the paths in
    ``BLOCK_PATH_SEQUENCES``.  Blocks that cannot be parsed emit a warning
    and are skipped so partial coverage beats a hard crash.

    Returns:
        List ordered by block index (0 = closest to input).
    """
    blocks = _find_block_container(model)
    if blocks is None:
        warnings.warn(
            "find_transformer_layers: could not locate block container. "
            "Check BLOCK_PATH_SEQUENCES or add the model's path manually.",
            stacklevel=2,
        )
        return []

    specs: List[TransformerLayerSpec] = []
    n_skipped = 0

    for idx, block in enumerate(blocks):
        # Find attention sub-module
        attn_module, attn_attr = _find_submodule_by_names(block, ATTN_NAMES)
        if attn_module is None:
            warnings.warn(f"Block {idx}: no attention sub-module found (tried {ATTN_NAMES})")
            n_skipped += 1
            continue

        attn_spec = _build_attn_spec(block, attn_module, attn_attr, model)
        if attn_spec is None:
            warnings.warn(f"Block {idx}: could not parse attention projections")
            n_skipped += 1
            continue

        # Find MLP sub-module
        mlp_module, mlp_attr = _find_submodule_by_names(block, MLP_NAMES)
        if mlp_module is None:
            warnings.warn(f"Block {idx}: no FFN sub-module found (tried {MLP_NAMES})")
            n_skipped += 1
            continue

        ffn_spec = _build_ffn_spec(mlp_module, mlp_attr)
        if ffn_spec is None:
            warnings.warn(f"Block {idx}: could not parse FFN layers")
            n_skipped += 1
            continue

        specs.append(TransformerLayerSpec(
            block_idx=idx, block=block,
            attn=attn_spec, ffn=ffn_spec,
        ))

    if n_skipped:
        warnings.warn(f"find_transformer_layers: skipped {n_skipped}/{len(blocks)} blocks")

    return specs


# ---------------------------------------------------------------------------
# 5. Importance scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_attention_heads(spec: AttentionSpec) -> torch.Tensor:
    """
    Return 1-D tensor of shape ``[num_heads]`` with L2-norm importance per head.

    Importance of head h = L2 norm of columns ``[h*head_dim : (h+1)*head_dim]``
    in ``out_proj.weight`` (shape ``[embed_dim, num_heads * head_dim]``).
    No forward pass required.
    """
    W = spec.out_proj.weight.data     # [embed_dim, num_heads * head_dim]
    hd = spec.head_dim
    # Reshape to [embed_dim, num_heads, head_dim] and norm over embed+head_dim dims
    return W.reshape(W.shape[0], spec.num_heads, hd).norm(dim=(0, 2))


@torch.no_grad()
def _score_ffn_neurons(spec: FFNSpec) -> torch.Tensor:
    """Return 1-D tensor of shape ``[intermediate_dim]`` with per-neuron importance."""
    if spec.ffn_type == "standard":
        # L2 norm of each input column of the down-projection
        return spec.fc_down.weight.data.norm(dim=0)   # [intermediate_dim]
    # SwiGLU: geometric-mean of gate × up × down norms (matches run_llm_experiment.py)
    return (
        spec.gate.weight.data.norm(dim=1)   # [intermediate_dim]
        * spec.up.weight.data.norm(dim=1)
        * spec.down.weight.data.norm(dim=0)
    )


# ---------------------------------------------------------------------------
# 6. Pruning helpers (private)
# ---------------------------------------------------------------------------

def _sparsity_to_n_keep_attn(spec: AttentionSpec, sparsity: float) -> int:
    """
    Convert sparsity fraction to integer keep-count, respecting GQA constraints.

    For GQA, n_keep must be a multiple of ``num_kv_heads`` so each KV group
    retains an equal number of Q-heads.
    """
    n_keep = max(1, round(spec.num_heads * (1.0 - sparsity)))
    if spec.is_gqa:
        n_keep = max(spec.num_kv_heads,
                     round(n_keep / spec.num_kv_heads) * spec.num_kv_heads)
    return min(n_keep, spec.num_heads)


def _sparsity_to_n_keep_ffn(spec: FFNSpec, sparsity: float) -> int:
    return max(1, round(spec.intermediate_dim * (1.0 - sparsity)))


def _select_heads_gqa(
    scores: torch.Tensor, n_keep: int, spec: AttentionSpec
) -> List[int]:
    """
    Choose n_keep Q-head indices while preserving equal representation per KV group.

    Within each KV group (slice of gqa_ratio Q-heads), keep the top-scoring
    n_keep // num_kv_heads heads.
    """
    grp = spec.gqa_ratio
    n_kv = spec.num_kv_heads
    n_per_grp = n_keep // n_kv
    keep: List[int] = []
    for g in range(n_kv):
        start = g * grp
        group_scores = scores[start: start + grp]
        local = group_scores.topk(n_per_grp, largest=True).indices.sort().values.tolist()
        keep.extend(start + i for i in local)
    return sorted(keep)


def _rebuild_linear(
    parent: nn.Module,
    attr: str,
    old: nn.Linear,
    in_f: int,
    out_f: int,
    weight_rows: Optional[torch.Tensor],   # None = keep all rows
    weight_cols: Optional[torch.Tensor],   # None = keep all cols
) -> nn.Linear:
    """
    Rebuild a Linear layer slicing rows and/or columns from the old weight.
    Writes the new module back to *parent* via setattr.
    """
    device = old.weight.device
    dtype  = old.weight.dtype
    new = nn.Linear(in_f, out_f, bias=old.bias is not None, device=device, dtype=dtype)
    with torch.no_grad():
        w = old.weight.data
        if weight_rows is not None:
            w = w[weight_rows]
        if weight_cols is not None:
            w = w[:, weight_cols]
        new.weight.copy_(w)
        if old.bias is not None:
            b = old.bias.data
            if weight_rows is not None:
                b = b[weight_rows]
            new.bias.copy_(b)
    setattr(parent, attr, new)
    return new


def _prune_attn_fused(
    spec: AttentionSpec, keep_idx: List[int], device: torch.device
) -> None:
    """Rebuild fused QKV and output projection keeping only *keep_idx* heads."""
    hd = spec.head_dim
    inner_dim = spec.num_heads * hd
    new_inner  = len(keep_idx) * hd

    # Build Q-block row indices on-device, then tile for K and V blocks
    base = torch.tensor(keep_idx, dtype=torch.long, device=spec.fused_qkv.weight.device)
    offsets = torch.arange(hd, dtype=torch.long, device=spec.fused_qkv.weight.device)
    q_rows = (base[:, None] * hd + offsets[None, :]).reshape(-1)   # [n_keep*hd]
    keep_rows = torch.cat([q_rows, q_rows + inner_dim, q_rows + 2 * inner_dim])
    keep_cols = q_rows

    embed_dim = spec.embed_dim
    new_qkv = _rebuild_linear(
        spec.attn_module, spec.fused_qkv_attr, spec.fused_qkv,
        embed_dim, 3 * new_inner, keep_rows, None,
    )
    new_out = _rebuild_linear(
        spec.attn_module, spec.out_proj_attr, spec.out_proj,
        new_inner, embed_dim, None, keep_cols,
    )
    spec.fused_qkv = new_qkv
    spec.out_proj   = new_out


def _prune_attn_separate(
    spec: AttentionSpec, keep_idx: List[int], device: torch.device
) -> None:
    """
    Rebuild separate Q/K/V and output projection.

    GQA: K and V are sized by num_kv_heads — they are NOT resized here.
    Only Q and out_proj change dimensions.
    """
    hd = spec.head_dim
    new_inner = len(keep_idx) * hd
    keep_rows = torch.tensor(
        [i for h in keep_idx for i in range(h * hd, (h + 1) * hd)],
        dtype=torch.long, device=spec.q.weight.device,
    )
    embed_dim = spec.embed_dim

    # Q: always resize
    new_q = _rebuild_linear(
        spec.attn_module, spec.q_attr, spec.q,
        embed_dim, new_inner, keep_rows, None,
    )
    spec.q = new_q

    if not spec.is_gqa:
        # Standard MHA: K and V share the same head structure as Q
        if spec.k is not None and spec.k_attr is not None:
            new_k = _rebuild_linear(
                spec.attn_module, spec.k_attr, spec.k,
                embed_dim, new_inner, keep_rows, None,
            )
            spec.k = new_k
        if spec.v is not None and spec.v_attr is not None:
            new_v = _rebuild_linear(
                spec.attn_module, spec.v_attr, spec.v,
                embed_dim, new_inner, keep_rows, None,
            )
            spec.v = new_v

    # out_proj: column indices correspond to Q-head outputs
    new_out = _rebuild_linear(
        spec.attn_module, spec.out_proj_attr, spec.out_proj,
        new_inner, embed_dim, None, keep_rows,
    )
    spec.out_proj = new_out


def _update_attn_metadata(spec: AttentionSpec, n_keep_heads: int) -> None:
    """Update head-count attributes on the live attn_module after resizing."""
    m = spec.attn_module
    for attr in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
        if hasattr(m, attr) and isinstance(getattr(m, attr), int):
            setattr(m, attr, n_keep_heads)

    new_inner = n_keep_heads * spec.head_dim
    for attr in ("attn_dim", "inner_dim", "all_head_size", "attention_inner_dim"):
        if hasattr(m, attr) and isinstance(getattr(m, attr), int):
            setattr(m, attr, new_inner)

    if spec.is_gqa:
        new_groups = n_keep_heads // spec.num_kv_heads
        for attr in ("num_key_value_groups", "num_heads_per_kv_group"):
            if hasattr(m, attr) and isinstance(getattr(m, attr), int):
                setattr(m, attr, new_groups)

    # Keep spec in sync so callers see consistent post-prune values.
    # model.config is updated centrally in transformer_structured_prune.
    spec.num_heads = n_keep_heads
    spec.gqa_ratio = max(1, n_keep_heads // spec.num_kv_heads)


def _prune_ffn_standard(spec: FFNSpec, n_keep: int) -> None:
    """Rebuild standard fc_up / fc_down keeping top-*n_keep* neurons."""
    scores   = _score_ffn_neurons(spec)
    keep_idx = scores.topk(n_keep, largest=True).indices.sort().values

    in_f  = spec.fc_up.in_features
    out_f = spec.fc_down.out_features

    new_up = _rebuild_linear(
        spec.mlp_module, spec.fc_up_attr, spec.fc_up,
        in_f, n_keep, keep_idx, None,
    )
    new_down = _rebuild_linear(
        spec.mlp_module, spec.fc_down_attr, spec.fc_down,
        n_keep, out_f, None, keep_idx,
    )
    spec.fc_up   = new_up
    spec.fc_down = new_down


def _prune_ffn_swiglu(spec: FFNSpec, n_keep: int) -> None:
    """Rebuild SwiGLU gate/up/down (and optional norm) keeping top-*n_keep* neurons."""
    scores   = _score_ffn_neurons(spec)
    keep_idx = scores.topk(n_keep, largest=True).indices.sort().values

    in_f  = spec.gate.in_features
    out_f = spec.down.out_features

    new_gate = _rebuild_linear(
        spec.mlp_module, spec.gate_attr, spec.gate, in_f, n_keep, keep_idx, None,
    )
    new_up = _rebuild_linear(
        spec.mlp_module, spec.up_attr,   spec.up,   in_f, n_keep, keep_idx, None,
    )
    new_down = _rebuild_linear(
        spec.mlp_module, spec.down_attr, spec.down, n_keep, out_f, None, keep_idx,
    )
    spec.gate = new_gate
    spec.up   = new_up
    spec.down = new_down

    # Optional intermediate LayerNorm (EVA02)
    if spec.norm is not None and spec.norm_attr is not None:
        old_norm = spec.norm
        new_norm = nn.LayerNorm(
            n_keep, eps=getattr(old_norm, "eps", 1e-6),
            device=old_norm.weight.device, dtype=old_norm.weight.dtype,
        )
        with torch.no_grad():
            new_norm.weight.copy_(old_norm.weight.data[keep_idx])
            new_norm.bias.copy_(old_norm.bias.data[keep_idx])
        setattr(spec.mlp_module, spec.norm_attr, new_norm)
        spec.norm = new_norm


# ---------------------------------------------------------------------------
# 7. Public: structural pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_attention_structural(
    spec: AttentionSpec,
    n_keep_heads: int,
    device: torch.device,
) -> None:
    """
    Physically resize attention projections in-place, keeping *n_keep_heads* heads.

    Selects the highest-importance heads by L2 norm of out_proj columns.
    After this call, ``spec.attn_module`` holds the rebuilt Linear layers and
    ``spec.num_heads`` is updated via ``_update_attn_metadata``.
    """
    n_keep_heads = max(1, min(n_keep_heads, spec.num_heads))
    scores = score_attention_heads(spec)

    if spec.is_gqa:
        keep_idx = _select_heads_gqa(scores, n_keep_heads, spec)
    else:
        keep_idx = scores.topk(n_keep_heads, largest=True).indices.sort().values.tolist()

    if spec.is_fused:
        _prune_attn_fused(spec, keep_idx, device)
    else:
        _prune_attn_separate(spec, keep_idx, device)

    _update_attn_metadata(spec, n_keep_heads)


@torch.no_grad()
def prune_ffn_structural(
    spec: FFNSpec,
    n_keep_neurons: int,
    device: torch.device,
) -> None:
    """
    Physically resize FFN projections in-place, keeping *n_keep_neurons* neurons.

    Supports both standard two-layer FFN and SwiGLU three-branch FFN.
    """
    n_keep_neurons = max(1, min(n_keep_neurons, spec.intermediate_dim))
    if spec.ffn_type == "standard":
        _prune_ffn_standard(spec, n_keep_neurons)
    else:
        _prune_ffn_swiglu(spec, n_keep_neurons)
    spec.intermediate_dim = n_keep_neurons  # keep spec in sync


# ---------------------------------------------------------------------------
# 8. Sensitivity scan (GPU-parallel, block-level)
# ---------------------------------------------------------------------------

def _sensitivity_worker(
    original_model: nn.Module,
    block_idx: int,
    component: str,          # "attention" | "ffn"
    sparsity: float,
    dataloader: dict,
    dense_acc: float,
    evaluate_fn: Callable,
    gpu_queue: queue.Queue,
    results: list,
    results_lock: threading.Lock,
    print_lock: threading.Lock,
    verbose: bool,
) -> None:
    """Worker: prune one block at one sparsity, evaluate, record drop."""
    gpu_id = gpu_queue.get()
    device = torch.device(f"cuda:{gpu_id}")
    try:
        model_copy = copy.deepcopy(original_model).to(device)
        specs = find_transformer_layers(model_copy)

        if block_idx >= len(specs):
            return

        spec = specs[block_idx]

        if component == "attention":
            n_keep = _sparsity_to_n_keep_attn(spec.attn, sparsity)
            prune_attention_structural(spec.attn, n_keep, device)
        else:
            n_keep = _sparsity_to_n_keep_ffn(spec.ffn, sparsity)
            prune_ffn_structural(spec.ffn, n_keep, device)

        acc  = evaluate_fn(model_copy, dataloader["test"], device)
        drop = acc - dense_acc

        with results_lock:
            results.append((block_idx, component, sparsity, drop))
        if verbose:
            with print_lock:
                print(
                    f"  [GPU {gpu_id}] block={block_idx:>3} {component:<10}"
                    f"  sp={sparsity:.2f}  drop={drop:+.2f}%"
                )
    except Exception as exc:
        with print_lock:
            print(
                f"  [GPU {gpu_id}] block={block_idx} {component} sp={sparsity:.2f}"
                f"  ERROR: {exc}"
            )
    finally:
        try:
            del model_copy
            torch.cuda.empty_cache()
        except Exception:
            pass
        gpu_queue.put(gpu_id)


def _build_sparsity_plan(
    results: list,
    n_blocks: int,
    sparsities: list,
    degradation_value: float,
) -> Dict[int, Dict[str, float]]:
    """
    Post-process scan results into per-block sparsity plan.

    For each (block, component), take the highest sparsity where
    ``|drop| <= degradation_value / 3``.  Falls back to best-accuracy
    sparsity if it exceeds -0.60 accuracy drop, else 0.0.
    """
    grouped: Dict[Tuple[int, str], List[Tuple[float, float]]] = defaultdict(list)
    for block_idx, component, sp, drop in results:
        grouped[(block_idx, component)].append((sp, drop))

    plan: Dict[int, Dict[str, float]] = {}
    threshold = degradation_value / 3.0

    for block_idx in range(n_blocks):
        plan[block_idx] = {}
        for component in ("attention", "ffn"):
            data = sorted(grouped.get((block_idx, component), []),
                          key=lambda x: x[0], reverse=True)  # high sparsity first
            best_sp = 0.0
            best_partial: Optional[Tuple[float, float]] = None
            for sp, drop in data:
                if abs(drop) <= threshold:
                    best_sp = sp
                    break
                if best_partial is None or drop > best_partial[1]:
                    best_partial = (sp, drop)
            if best_sp == 0.0 and best_partial is not None and best_partial[1] > -0.60:
                best_sp = best_partial[0]
            if best_sp > 0.0:
                plan[block_idx][component] = best_sp

    return plan


def transformer_sensitivity_scan(
    model: nn.Module,
    dataloader: dict,
    dense_acc: float,
    evaluate_fn: Callable,
    scan_step: float = 0.1,
    scan_start: float = 0.1,
    scan_end: float = 0.9,
    degradation_value: float = 1.2,
    safety_net_gb: float = 2.0,
    activation_multiplier: float = 1.5,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    GPU-parallel block-level sensitivity scan for Transformer pruning.

    For every (block, component, sparsity) triplet, deep-copies the model,
    prunes that single block/component at that sparsity, evaluates accuracy,
    and records the drop.

    Args:
        model:               The model to scan (not modified).
        dataloader:          Dict with at least a ``"test"`` key.
        dense_acc:           Baseline accuracy (%), used to compute drop.
        evaluate_fn:         Callable ``(model, test_loader, device) -> float (%)``
        scan_step / start / end:  Sparsity sweep parameters.
        degradation_value:   Max acceptable accuracy drop (%) ÷ 3 = threshold.
        safety_net_gb:       VRAM headroom reserved per GPU.
        activation_multiplier: Multiplier on measured model VRAM for activations.
        verbose:             Print per-job progress.

    Returns:
        ``{block_idx: {"attention": sparsity, "ffn": sparsity}}``
        Only blocks / components with sparsity > 0.0 are included.
    """
    specs = find_transformer_layers(model)
    if not specs:
        warnings.warn("transformer_sensitivity_scan: no transformer layers found. "
                      "Returning empty plan.")
        return {}

    n_blocks   = len(specs)
    sparsities = list(np.arange(scan_start, scan_end + 1e-9, scan_step).round(4))
    all_jobs   = [
        (block_idx, component, sp)
        for sp in sparsities
        for block_idx in range(n_blocks)
        for component in ("attention", "ffn")
    ]
    total_jobs = len(all_jobs)

    if not torch.cuda.is_available():
        # ── Serial fallback ──────────────────────────────────────────────
        if verbose:
            print(f"[transformer sensitivity scan] no CUDA — serial fallback")
            print(f"  blocks={n_blocks}  sparsities={len(sparsities)}  jobs={total_jobs}")
        results: list = []
        device = torch.device("cpu")
        for block_idx, component, sp in all_jobs:
            model_copy = copy.deepcopy(model).to(device)
            spec_copy  = find_transformer_layers(model_copy)[block_idx]
            if component == "attention":
                n_keep = _sparsity_to_n_keep_attn(spec_copy.attn, sp)
                prune_attention_structural(spec_copy.attn, n_keep, device)
            else:
                n_keep = _sparsity_to_n_keep_ffn(spec_copy.ffn, sp)
                prune_ffn_structural(spec_copy.ffn, n_keep, device)
            acc  = evaluate_fn(model_copy, dataloader["test"], device)
            drop = acc - dense_acc
            results.append((block_idx, component, sp, drop))
            if verbose:
                print(f"  block={block_idx} {component} sp={sp:.2f} drop={drop:+.2f}%")
            del model_copy
        return _build_sparsity_plan(results, n_blocks, sparsities, degradation_value)

    # ── GPU pool ────────────────────────────────────────────────────────
    gpu_list = _nvidia_smi_free_gb()
    if not gpu_list:
        warnings.warn("nvidia-smi unavailable; using GPU 0 with 1 worker")
        gpu_list = [(0, 8.0)]

    gpu_queue, total_slots, _ = _build_gpu_worker_pool(
        gpu_list, model, safety_net_gb, activation_multiplier,
        verbose, "transformer sensitivity scan",
    )
    if verbose:
        print(f"  Total slots: {total_slots}  |  jobs: {total_jobs}"
              f"  ({n_blocks} blocks × {len(sparsities)} sparsities × 2 components)")

    results_lock = threading.Lock()
    print_lock   = threading.Lock()
    results: list = []

    with ThreadPoolExecutor(max_workers=total_slots) as pool:
        futures = [
            pool.submit(
                _sensitivity_worker,
                model, block_idx, component, sp,
                dataloader, dense_acc, evaluate_fn,
                gpu_queue, results, results_lock, print_lock, verbose,
            )
            for block_idx, component, sp in all_jobs
        ]
        for f in as_completed(futures):
            exc = f.exception()
            if exc is not None and verbose:
                with print_lock:
                    print(f"  [worker exception] {exc}")

    return _build_sparsity_plan(results, n_blocks, sparsities, degradation_value)


# ---------------------------------------------------------------------------
# 9. Apply sparsity plan
# ---------------------------------------------------------------------------

def _update_model_config(
    model: nn.Module,
    cfg_updates: Dict[int, Dict[str, int]],
) -> None:
    """
    Propagate surviving dimension counts into *model.config* (HF models).

    Called once after all blocks are pruned so that ``save_pretrained`` /
    ``from_pretrained`` serialises the correct shapes.

    For heterogeneous pruning (different sparsity per block) the minimum
    surviving count is used and a warning is emitted — the resulting config
    is not sufficient for a faithful ``from_pretrained`` round-trip.
    """
    cfg = getattr(model, "config", None)
    if cfg is None or not cfg_updates:
        return

    # ── Attention heads ──────────────────────────────────────────────────
    attn_keeps = [v["attention"] for v in cfg_updates.values() if "attention" in v]
    if attn_keeps:
        min_heads = min(attn_keeps)
        for attr in ("num_attention_heads", "num_heads"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, min_heads)
        if len(set(attn_keeps)) > 1:
            warnings.warn(
                f"Heterogeneous attention sparsity detected: blocks have different "
                f"surviving head counts {sorted(set(attn_keeps))}. "
                f"model.config.num_attention_heads set to {min_heads} (minimum). "
                f"save_pretrained / from_pretrained may not reconstruct correctly.",
                stacklevel=3,
            )

    # ── FFN intermediate size ────────────────────────────────────────────
    ffn_keeps = [v["ffn"] for v in cfg_updates.values() if "ffn" in v]
    if ffn_keeps:
        min_intermediate = min(ffn_keeps)
        for attr in ("intermediate_size", "ffn_dim", "n_inner", "d_ffn"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, min_intermediate)


@torch.no_grad()
def transformer_structured_prune(
    model: nn.Module,
    sparsity_plan: Dict[int, Dict[str, float]],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Apply *sparsity_plan* to *model* in-place.

    Args:
        model:         The model to prune (modified in-place).
        sparsity_plan: ``{block_idx: {"attention": float, "ffn": float}}``.
                       Missing keys → no pruning for that block/component.
        device:        Target device; defaults to the model's current device.

    Returns:
        The pruned model (same object as *model*).
    """
    if device is None:
        device = next(model.parameters()).device

    specs = find_transformer_layers(model)
    if not specs:
        warnings.warn("transformer_structured_prune: no transformer layers found.")
        return model

    cfg_updates: Dict[int, Dict[str, int]] = {}

    for spec in specs:
        plan = sparsity_plan.get(spec.block_idx, {})

        attn_sp = plan.get("attention", 0.0)
        if attn_sp > 0.0:
            n_keep = _sparsity_to_n_keep_attn(spec.attn, attn_sp)
            prune_attention_structural(spec.attn, n_keep, device)
            cfg_updates.setdefault(spec.block_idx, {})["attention"] = n_keep

        ffn_sp = plan.get("ffn", 0.0)
        if ffn_sp > 0.0:
            n_keep = _sparsity_to_n_keep_ffn(spec.ffn, ffn_sp)
            prune_ffn_structural(spec.ffn, n_keep, device)
            cfg_updates.setdefault(spec.block_idx, {})["ffn"] = n_keep

    _update_model_config(model, cfg_updates)

    return model
