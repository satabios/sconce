import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm
import copy
import torch.nn as nn
from collections import OrderedDict, defaultdict
from typing import Union, List

try:
    import torch_pruning as tp
    _TORCH_PRUNING_AVAILABLE = True
except ImportError:
    tp = None
    _TORCH_PRUNING_AVAILABLE = False


def _nvidia_smi_free_gb() -> list[tuple[int, float]]:
    """Return [(gpu_id, free_gb), ...] via nvidia-smi. Empty list if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        result = []
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                result.append((int(parts[0].strip()), int(parts[1].strip()) / 1024))
        return result
    except Exception:
        return []


def _measure_model_vram_gb(model, activation_multiplier: float = 1.5) -> float:
    """Estimate model VRAM footprint in GB × multiplier (parameter + buffer byte count)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buf_bytes) / (1024 ** 3) * activation_multiplier


def _build_gpu_worker_pool(
    gpu_list: list[tuple[int, float]],
    model: nn.Module,
    safety_net_gb: float,
    activation_multiplier: float,
    verbose: bool,
    label: str,
) -> tuple[queue.Queue, int, float]:
    """Build a GPU worker queue from *gpu_list*.

    Allocates ``max(1, floor((free_gb - safety_net_gb) / model_vram_gb))`` workers
    per GPU.  Prints the allocation table when *verbose* is True.

    Returns:
        (gpu_queue, total_slots, model_vram_gb)
    """
    model_vram_gb = _measure_model_vram_gb(model, activation_multiplier)
    gpu_queue: queue.Queue = queue.Queue()
    alloc_rows = []
    for gpu_id, free_gb in gpu_list:
        n_workers = max(1, int((free_gb - safety_net_gb) / model_vram_gb))
        for _ in range(n_workers):
            gpu_queue.put(gpu_id)
        alloc_rows.append((gpu_id, free_gb, n_workers))
    total_slots = gpu_queue.qsize()
    if verbose:
        print(f"\n[{label}]  model_vram={model_vram_gb:.2f} GB  safety_net={safety_net_gb} GB")
        print(f"  {'GPU':>4}  {'free_GB':>8}  {'workers':>8}")
        for gpu_id, free_gb, nw in alloc_rows:
            print(f"  {gpu_id:>4}  {free_gb:>8.1f}  {nw:>8}")
    return gpu_queue, total_slots, model_vram_gb


def _collect_conv_bn(
    model: nn.Module,
) -> tuple[list[nn.Conv2d], list[nn.BatchNorm2d]]:
    """Return (all_convs, all_bns) collected in traversal order from *model*."""
    all_convs: list[nn.Conv2d] = []
    all_bns: list[nn.BatchNorm2d] = []

    def _walk(obj):
        if isinstance(obj, nn.Conv2d):
            all_convs.append(obj)
        elif isinstance(obj, nn.BatchNorm2d):
            all_bns.append(obj)
        elif isinstance(obj, list):
            for child in obj:
                _walk(child)
        elif isinstance(obj, OrderedDict):
            for child in obj.values():
                _walk(child)
        elif hasattr(obj, "children"):
            for child in obj.children():
                _walk(child)

    _walk(model)
    return all_convs, all_bns


# Module types eligible for structured (CWP-style) pruning.
# Covers CNN (Conv1d/2d/3d) and Transformer (Linear) layers.
PRUNABLE_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _get_example_inputs(dataloader, device=None):
    """Return a batch-size-1 example input from *dataloader*.

    Handles three common batch formats:
    - ``(Tensor, label)``  — standard classification / image
    - ``dict``             — HuggingFace-style (input_ids, attention_mask, …)
    - bare ``Tensor``      — unlabelled datasets

    Returns a ``Tensor`` or ``dict[str, Tensor]``.
    """
    batch = next(iter(dataloader))

    if isinstance(batch, dict):
        x = {k: v[:1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if device is not None:
            x = {k: v.to(device) for k, v in x.items()}
        return x

    if isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch

    x = x[:1]
    if device is not None:
        x = x.to(device)
    return x


def _structured_zero_prune(module: nn.Module, sparsity: float) -> None:
    """Zero out the lowest-importance output rows/channels of a prunable module.

    Works for **any** ``nn.Linear``, ``nn.Conv1d/2d/3d`` — no dependency graph
    needed and no dimension changes, so the model's forward pass stays valid.
    This makes it suitable for Transformer layers (Q/K/V/O projections, FFN
    up/down projections) where explicit coupling rules are hard to infer.

    Importance = L2 norm of each output unit across all input dimensions.

    Args:
        module:   ``nn.Linear`` or ``nn.Conv*`` instance.
        sparsity: fraction [0, 1] of outputs to zero.
    """
    if not isinstance(module, PRUNABLE_MODULES):
        return
    with torch.no_grad():
        w = module.weight                              # [out, in, ...]
        importance = w.view(w.shape[0], -1).norm(dim=1)
        n_prune = round(w.shape[0] * sparsity)
        if n_prune <= 0:
            return
        prune_idx = importance.argsort()[:n_prune]    # lowest importance first
        module.weight[prune_idx] = 0.0
        if getattr(module, 'bias', None) is not None:
            module.bias[prune_idx] = 0.0


def _cwp_prune_module(model, current_module, sparsity, example_inputs):
    """Apply structured pruning to *current_module* inside *model*.

    Tries ``torch_pruning.MetaPruner`` first (handles dependency propagation
    automatically for both CNN and Transformer graphs).  Falls back to
    ``_structured_zero_prune`` when torch_pruning is unavailable — safe for
    all architectures at the cost of not reducing layer dimensions.
    """
    if _TORCH_PRUNING_AVAILABLE:
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            pruning_ratio=0,
            pruning_ratio_dict={current_module: sparsity},
        )
        pruner.step()
    else:
        _structured_zero_prune(current_module, sparsity)


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

        if _TORCH_PRUNING_AVAILABLE:
            pruner = tp.pruner.MetaPruner(
                self.model,
                example_inputs,
                importance=tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
                pruning_ratio=0,
                pruning_ratio_dict=ratio_dict,
                round_to=8,
            )
            pruner.step()
        else:
            # torch_pruning not available: apply structured zeroing per module
            for module, sparsity in ratio_dict.items():
                _structured_zero_prune(module, sparsity)
