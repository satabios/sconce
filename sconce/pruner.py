import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import types
import torch.nn as nn
from collections import OrderedDict, defaultdict
from typing import Union, List
import torch_pruning as tp

class prune:

    def get_input_channel_importance_channel(self, weight, dim=1):
        in_channels = weight.shape[dim]
        importances = []
        for i_c in range(in_channels):
            if dim == 1:
                channel_weight = weight.detach()[:, i_c]
            else:
                channel_weight = weight.detach()[i_c]
            importance = torch.norm(channel_weight)
            importances.append(importance.view(1))
        return torch.cat(importances)

    def get_attention_head_importance(self, weights, num_heads):
        """Compute per-head importance from weight tensors of attention layers.

        Supports both fused QKV (single tensor, e.g. timm/torchvision) and
        separate Q/K/V (list of tensors, e.g. HuggingFace).

        Args:
            weights: single tensor of shape (3*num_heads*head_dim, embed_dim) for fused QKV,
                     or list/tuple of tensors [Q, K, V] each (num_heads*head_dim, embed_dim).
            num_heads: number of attention heads.

        Returns:
            Tensor of shape (num_heads,) with L2-norm importance per head.
        """
        if isinstance(weights, (list, tuple)):
            # Separate Q, K, V path (HuggingFace)
            head_importances = []
            for w in weights:
                head_dim = w.shape[0] // num_heads
                w_reshaped = w.detach().reshape(num_heads, -1)
                head_importances.append(torch.norm(w_reshaped, dim=1))
            return torch.stack(head_importances).mean(dim=0)
        else:
            # Fused QKV path (timm, torchvision in_proj_weight)
            head_dim = weights.shape[0] // (3 * num_heads)
            w = weights.detach().reshape(3 * num_heads, -1)
            # Compute L2 norm per row, then average every 3 (Q, K, V for same head)
            per_row_norm = torch.norm(w, dim=1)  # (3*num_heads,)
            # Reshape to (3, num_heads) and average across Q, K, V
            return per_row_norm.reshape(3, num_heads).mean(dim=0)

    def _detect_vit_config(self, model):
        """Auto-detect ViT framework and return pruning configuration.

        Scans model modules to identify timm, HuggingFace, or torchvision ViT
        architectures and builds the necessary metadata for Torch-Pruning.

        Returns:
            dict with keys:
                - 'framework': 'timm' | 'huggingface' | 'torchvision' | None
                - 'num_heads': dict mapping attention modules -> num_heads
                - 'ignored_layers': list of modules to skip (classifier head)
                - 'unwrapped_parameters': list of (Parameter, int) tuples for pruning dim
                - 'prunable_modules': list of (name, module) to iterate in sensitivity_scan
                - 'attention_modules': list of (name, attn_parent_module) for head pruning
        """
        framework = None
        num_heads = {}
        ignored_layers = []
        unwrapped_parameters = []
        prunable_modules = []
        attention_modules = []

        for name, module in model.named_modules():
            # timm: Attention class with fused 'qkv' Linear
            if hasattr(module, 'qkv') and hasattr(module, 'proj') and hasattr(module, 'num_heads'):
                framework = 'timm'
                num_heads[module.qkv] = module.num_heads
                attention_modules.append((name, module))

            # HuggingFace: ViTSelfAttention with separate query/key/value
            elif hasattr(module, 'query') and hasattr(module, 'key') and hasattr(module, 'value'):
                if hasattr(module, 'num_attention_heads'):
                    framework = 'huggingface'
                    num_heads[module.query] = module.num_attention_heads
                    num_heads[module.key] = module.num_attention_heads
                    num_heads[module.value] = module.num_attention_heads
                    attention_modules.append((name, module))

            # torchvision: nn.MultiheadAttention
            # NOTE: For torchvision, DG cascades embed_dim globally across ALL MHAs,
            # so we only store ONE representative MHA in attention_modules.
            # All MHAs still go into num_heads for MetaPruner compatibility.
            elif isinstance(module, nn.MultiheadAttention):
                framework = 'torchvision'
                num_heads[module] = module.num_heads
                if not any(isinstance(m, nn.MultiheadAttention) for _, m in attention_modules):
                    attention_modules.append((name, module))

            # Detect classifier heads to ignore (must be final classifier, not MLP fc1/fc2)
            is_classifier = (
                isinstance(module, nn.Linear) and
                (name.endswith('.head') or name == 'head' or
                 name.endswith('.classifier') or name == 'classifier' or
                 name.endswith('.fc') or name == 'fc')
            )
            if is_classifier:
                ignored_layers.append(module)

        # Collect unwrapped parameters (position embeddings, class tokens)
        for name, param in model.named_parameters():
            if any(kw in name.lower() for kw in [
                'pos_embed', 'position_embed', 'pos_embedding',
                'cls_token', 'class_token',
            ]):
                unwrapped_parameters.append((param, param.dim() - 1))

        # Classify modules for sensitivity_scan:
        # - MLP intermediate layers (fc1): safe to prune per-module
        # - Attention output projections, fc2, norms: affect embed_dim, skip
        # - Attention QKV: use head pruning (separate path via attention_modules)
        # - Conv2d (patch embed etc.): typically affects embed_dim, skip
        #
        # For timm: skip .proj, .fc2, .qkv
        # For HF: skip .query, .key, .value, .output.dense (attn out + MLP fc2)
        #          but keep .intermediate.dense (MLP fc1)
        # For torchvision: skip .out_proj; also skip MLP fc2 via dimension check
        skip_suffixes = {
            'timm': ['.proj', '.fc2', '.qkv'],
            'huggingface': ['.query', '.key', '.value', '.output.dense'],
            'torchvision': ['.out_proj'],
            None: [],
        }

        # Patch embed Conv2d layers affect embed_dim globally — always skip
        patch_embed_keywords = ['conv_proj', 'patch_embed', 'patch_embeddings']

        for name, module in model.named_modules():
            if any(name.endswith(s) for s in skip_suffixes.get(framework, [])):
                continue
            # Skip patch embed Conv2d layers
            if isinstance(module, nn.Conv2d) and any(kw in name for kw in patch_embed_keywords):
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module not in ignored_layers:
                    # For ViT architectures, only include MLP intermediate (expanding) layers
                    # Skip Linear layers that project back to embed_dim (fc2 equivalents)
                    if framework and isinstance(module, nn.Linear):
                        # MLP intermediate layers expand: out_features > in_features
                        # fc2/proj layers contract or keep same: out_features <= in_features
                        if module.out_features <= module.in_features:
                            continue
                    prunable_modules.append((name, module))

        return {
            'framework': framework,
            'num_heads': num_heads if num_heads else None,
            'ignored_layers': ignored_layers,
            'unwrapped_parameters': unwrapped_parameters if unwrapped_parameters else None,
            'prunable_modules': prunable_modules,
            'attention_modules': attention_modules,
        }

    def _build_metapruner_kwargs(self, vit_config):
        """Build the extra keyword arguments for MetaPruner when pruning a ViT."""
        kwargs = {}
        if vit_config is None:
            return kwargs
        if vit_config['num_heads']:
            kwargs['num_heads'] = vit_config['num_heads']
            # Only prune entire heads (not head_dim) to avoid dimension
            # mismatch with residual connections in transformers.
            kwargs['prune_num_heads'] = True
            kwargs['prune_head_dims'] = False
        if vit_config['unwrapped_parameters']:
            kwargs['unwrapped_parameters'] = vit_config['unwrapped_parameters']
        if vit_config['ignored_layers']:
            kwargs['ignored_layers'] = vit_config['ignored_layers']
        return kwargs

    def _update_head_counts_after_pruning(self, model, pruner, framework):
        """After MetaPruner.step(), sync num_heads/head_dim on attention modules."""
        if framework == 'timm':
            for _, module in model.named_modules():
                if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                    new_heads = pruner.num_heads.get(module.qkv, module.num_heads)
                    module.num_heads = new_heads
                    module.head_dim = module.qkv.out_features // (3 * new_heads)
        elif framework == 'torchvision':
            for _, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    new_heads = pruner.num_heads.get(module, module.num_heads)
                    module.num_heads = new_heads
        elif framework == 'huggingface':
            for _, module in model.named_modules():
                if hasattr(module, 'query') and hasattr(module, 'num_attention_heads'):
                    new_heads = pruner.num_heads.get(module.query, module.num_attention_heads)
                    module.num_attention_heads = new_heads
                    module.attention_head_size = module.query.out_features // new_heads
                    module.all_head_size = module.query.out_features

    @staticmethod
    def _patched_timm_attn_forward(self, x, attn_mask=None):
        """Patched forward for timm Attention that uses -1 reshape.

        timm's original forward uses C from input shape to reshape attention
        output, but after head pruning num_heads*head_dim != embed_dim.
        This version uses -1 to infer the correct reshape dimension.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _patch_timm_attention(self, model):
        """Patch all timm Attention modules to use -1 reshape for pruning compat."""
        for _, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'proj') and hasattr(module, 'num_heads'):
                module.forward = types.MethodType(self._patched_timm_attn_forward, module)

    def _prune_attention_heads(self, model, example_inputs, attn_module, name,
                               sparsity, framework, vit_config):
        """Prune attention heads using DependencyGraph.

        Removes entire heads from the attention module.

        - timm: prunes QKV output and proj input, keeps embed_dim (proj output)
          unchanged. Requires _patch_timm_attention() before calling.
        - huggingface: prunes query output, DG cascades to key/value/dense.
          Keeps embed_dim unchanged.
        - torchvision: prunes MHA out_channels which cascades through the entire
          model's embed_dim. Updates model.hidden_dim and all MHA num_heads.
          Only call once — DG cascades to ALL MHA layers.

        Args:
            model: the model to prune
            example_inputs: example input tensor for DG tracing
            attn_module: the attention parent module (timm Attention, HF ViTSelfAttention, or nn.MHA)
            name: dotted module name
            sparsity: fraction of heads to remove (0.0 to 1.0)
            framework: 'timm', 'huggingface', or 'torchvision'
            vit_config: ViT config dict from _detect_vit_config
        """
        unwrapped = vit_config['unwrapped_parameters'] if vit_config and vit_config['unwrapped_parameters'] else None
        # Wrap as tuple so DG._trace does model(*(tensor,)) = model(tensor) correctly
        if not isinstance(example_inputs, tuple):
            example_inputs = (example_inputs,)
        DG = tp.DependencyGraph().build_dependency(
            model, example_inputs=example_inputs,
            unwrapped_parameters=unwrapped,
            ignored_params=[],
        )

        if framework == 'timm':
            num_heads = attn_module.num_heads
            head_dim = attn_module.head_dim
            qkv = attn_module.qkv
            importance = self.get_attention_head_importance(qkv.weight, num_heads)
            n_prune = max(1, int(round(num_heads * sparsity)))
            if n_prune >= num_heads:
                n_prune = num_heads - 1
            head_indices = torch.argsort(importance)[:n_prune]
            # Build fused QKV pruning indices: [Q_h0, Q_h1, ..., K_h0, ..., V_h0, ...]
            prune_idxs = []
            for h in head_indices:
                for qkv_offset in range(3):
                    start = qkv_offset * num_heads * head_dim + h.item() * head_dim
                    prune_idxs.extend(range(start, start + head_dim))
            group = DG.get_pruning_group(qkv, tp.prune_linear_out_channels, idxs=prune_idxs)
            if group is not None:
                group.prune()
            attn_module.num_heads = num_heads - n_prune

        elif framework == 'huggingface':
            num_heads = attn_module.num_attention_heads
            head_dim = attn_module.query.out_features // num_heads
            importance = self.get_attention_head_importance(
                [attn_module.query.weight, attn_module.key.weight, attn_module.value.weight],
                num_heads
            )
            n_prune = max(1, int(round(num_heads * sparsity)))
            if n_prune >= num_heads:
                n_prune = num_heads - 1
            head_indices = torch.argsort(importance)[:n_prune]
            prune_idxs = []
            for h in head_indices:
                prune_idxs.extend(range(h.item() * head_dim, (h.item() + 1) * head_dim))
            group = DG.get_pruning_group(attn_module.query, tp.prune_linear_out_channels, idxs=prune_idxs)
            if group is not None:
                group.prune()
            attn_module.num_attention_heads = num_heads - n_prune
            attn_module.attention_head_size = head_dim
            attn_module.all_head_size = attn_module.query.out_features

        elif framework == 'torchvision':
            # For torchvision MHA, pruning out_channels cascades through the
            # entire model (embed_dim changes globally). Only call this ONCE —
            # DG handles all MHA layers together.
            num_heads = attn_module.num_heads
            head_dim = attn_module.embed_dim // num_heads
            importance = self.get_attention_head_importance(attn_module.in_proj_weight, num_heads)
            n_prune = max(1, int(round(num_heads * sparsity)))
            if n_prune >= num_heads:
                n_prune = num_heads - 1
            head_indices = torch.argsort(importance)[:n_prune]
            prune_idxs = []
            for h in head_indices:
                prune_idxs.extend(range(h.item() * head_dim, (h.item() + 1) * head_dim))
            group = DG.get_pruning_group(attn_module, tp.prune_multihead_attention_out_channels, idxs=prune_idxs)
            if group is not None:
                group.prune()
            # Update model-level hidden_dim (used in _process_input reshape)
            if hasattr(model, 'hidden_dim'):
                model.hidden_dim = attn_module.embed_dim
            # Update num_heads on ALL MHAs (DG cascaded embed_dim globally)
            new_heads = num_heads - n_prune
            for _, mod in model.named_modules():
                if isinstance(mod, nn.MultiheadAttention):
                    mod.num_heads = new_heads
                    mod.head_dim = mod.embed_dim // new_heads

    def _prune_with_dependency_graph(self, model, example_inputs, module, name, sparsity, vit_config):
        """Prune using custom importance scores + DependencyGraph for manual control.

        This is the 'custom importance' path — alternative to MetaPruner.
        """
        unwrapped = None
        if vit_config and vit_config['unwrapped_parameters']:
            unwrapped = vit_config['unwrapped_parameters']
        DG = tp.DependencyGraph().build_dependency(
            model, example_inputs=example_inputs,
            unwrapped_parameters=unwrapped,
            ignored_params=[],
        )
        framework = vit_config['framework'] if vit_config else None
        group = None

        if isinstance(module, nn.Conv2d):
            importance = self.get_input_channel_importance_channel(module.weight, dim=0)
            n_channels = module.out_channels
            n_prune = int(round(n_channels * sparsity))
            if n_prune == 0:
                return
            prune_indices = torch.argsort(importance)[:n_prune]
            group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=prune_indices.tolist())

        elif isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = dict(model.named_modules()).get(parent_name)

            if framework == 'timm' and hasattr(parent, 'num_heads') and name.endswith('.qkv'):
                num_heads = parent.num_heads
                importance = self.get_attention_head_importance(module.weight, num_heads)
                n_prune = max(1, int(round(num_heads * sparsity)))
                if n_prune >= num_heads:
                    n_prune = num_heads - 1
                head_indices = torch.argsort(importance)[:n_prune]
                head_dim = module.out_features // (3 * num_heads)
                prune_idxs = []
                for h in head_indices:
                    for qkv_offset in range(3):
                        start = qkv_offset * num_heads * head_dim + h.item() * head_dim
                        prune_idxs.extend(range(start, start + head_dim))
                group = DG.get_pruning_group(module, tp.prune_linear_out_channels, idxs=prune_idxs)

            elif framework == 'huggingface' and hasattr(parent, 'num_attention_heads') and name.endswith('.query'):
                num_heads = parent.num_attention_heads
                importance = self.get_attention_head_importance(
                    [parent.query.weight, parent.key.weight, parent.value.weight], num_heads
                )
                n_prune = max(1, int(round(num_heads * sparsity)))
                if n_prune >= num_heads:
                    n_prune = num_heads - 1
                head_indices = torch.argsort(importance)[:n_prune]
                head_dim = parent.query.out_features // num_heads
                prune_idxs = []
                for h in head_indices:
                    prune_idxs.extend(range(h.item() * head_dim, (h.item() + 1) * head_dim))
                group = DG.get_pruning_group(parent.query, tp.prune_linear_out_channels, idxs=prune_idxs)

            else:
                # Regular Linear (MLP layers)
                importance = self.get_input_channel_importance_channel(module.weight, dim=0)
                n_channels = module.out_features
                n_prune = int(round(n_channels * sparsity))
                if n_prune == 0:
                    return
                prune_indices = torch.argsort(importance)[:n_prune]
                group = DG.get_pruning_group(module, tp.prune_linear_out_channels, idxs=prune_indices.tolist())

        elif isinstance(module, nn.MultiheadAttention):
            num_heads = module.num_heads
            importance = self.get_attention_head_importance(module.in_proj_weight, num_heads)
            n_prune = max(1, int(round(num_heads * sparsity)))
            if n_prune >= num_heads:
                n_prune = num_heads - 1
            head_indices = torch.argsort(importance)[:n_prune]
            head_dim = module.embed_dim // num_heads
            prune_idxs = []
            for h in head_indices:
                prune_idxs.extend(range(h.item() * head_dim, (h.item() + 1) * head_dim))
            group = DG.get_pruning_group(module, tp.prune_multihead_attention_out_channels, idxs=prune_idxs)

        if group is not None:
            group.prune()

    def _scan_single_module(self, original_model, name, sparsities, dense_model_accuracy,
                            example_inputs, vit_config, verbose, i_layer, total_layers,
                            is_attention=False, framework=None):
        """Scan sparsity levels for a single module and return (name, best_sparsity).

        Args:
            is_attention: if True, use DG-based head pruning instead of MetaPruner.
            framework: ViT framework string (needed for attention path).
        """
        accuracy = []
        desc = f"scanning {i_layer}/{total_layers} - {name}" if verbose else None
        picker = tqdm(sparsities, desc=desc) if verbose else sparsities

        for sparsity in picker:
            # Reset model to original state at each sparsity step
            self.model = copy.deepcopy(original_model)

            if self.prune_mode == "CWP":
                if is_attention and framework:
                    # Attention head pruning via DependencyGraph
                    current_vit_config = self._detect_vit_config(self.model)
                    # Find the attention module in the fresh copy
                    attn_module = dict(self.model.named_modules())[name]
                    if framework == 'timm':
                        self._patch_timm_attention(self.model)
                    self._prune_attention_heads(
                        self.model, example_inputs, attn_module, name,
                        sparsity, framework, current_vit_config,
                    )
                elif self.use_custom_importance:
                    # Path A: Custom importance + DependencyGraph for MLP/Conv modules
                    current_module = dict(self.model.named_modules())[name]
                    current_vit_config = self._detect_vit_config(self.model) if self.attention_heads else None
                    self._prune_with_dependency_graph(
                        self.model, example_inputs, current_module, name, sparsity, current_vit_config
                    )
                else:
                    # Path B: MetaPruner (automated) for MLP/Conv modules
                    current_module = dict(self.model.named_modules())[name]
                    pruner_kwargs = {
                        'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
                        'pruning_ratio': 0,
                        'pruning_ratio_dict': {current_module: sparsity},
                    }
                    if vit_config and vit_config['framework']:
                        current_vit_config = self._detect_vit_config(self.model)
                        pruner_kwargs.update(self._build_metapruner_kwargs(current_vit_config))

                    pruner = tp.pruner.MetaPruner(self.model, example_inputs, **pruner_kwargs)
                    pruner.step()

                    if vit_config and vit_config['framework']:
                        self._update_head_counts_after_pruning(self.model, pruner, vit_config['framework'])

            elif self.prune_mode == "GMP":
                # GMP not applicable for attention heads
                if is_attention:
                    continue
                current_param = dict(self.model.named_parameters())[name]
                sparse_list = np.zeros(total_layers)
                sparse_list[i_layer] = sparsity
                param_names = [n for n, p in original_model.named_parameters() if p.dim() > 1]
                local_sparsity_dict = dict(zip(param_names, sparse_list))
                self.GMP_Pruning(prune_dict=local_sparsity_dict)
                self.callbacks = [lambda: self.GMP_apply()]

            # Evaluate the pruned model
            acc = self.evaluate(Tqdm=False) - dense_model_accuracy
            if abs(acc) <= (self.degradation_value) / 3:
                self.model = copy.deepcopy(original_model)
                return name, sparsity
            elif sparsity == sparsities[-1]:  # Last sparsity step
                self.model = copy.deepcopy(original_model)
                if accuracy:
                    best_sparsity = sparsities[np.argmax(accuracy)]
                    return name, best_sparsity if np.max(accuracy) > -0.60 else 0.0
                else:
                    return name, 0.0
            else:
                accuracy.append(acc)

        # Reset model after scanning all sparsities
        self.model = copy.deepcopy(original_model)
        return name, 0.0

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

        For ViT models (when attention_heads=True), uses a two-path approach:
        - MLP intermediate layers (fc1): scanned via MetaPruner per-module
        - Attention modules: scanned via DG-based head pruning with patched forward

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

        # Detect ViT configuration if attention_heads mode is enabled
        vit_config = self._detect_vit_config(original_model) if self.attention_heads else None
        original_prune_mode = self.prune_mode

        # Build the list of modules to scan and example inputs
        example_inputs = None

        if self.prune_mode == "CWP":
            model_device = next(original_model.parameters()).device
            example_inputs = next(iter(self.dataloader['test']))[0][:1, :].to(model_device)

            if vit_config and vit_config['framework']:
                # ViT mode: prunable_modules has MLP-safe layers only
                named_all_weights = vit_config['prunable_modules']
                attention_modules = vit_config['attention_modules']
                framework = vit_config['framework']
            else:
                # Original Conv2d-only path
                named_all_weights = [
                    (name, module)
                    for name, module in original_model.named_modules()
                    if isinstance(module, nn.Conv2d)
                ]
                attention_modules = []
                framework = None
        elif self.prune_mode == "GMP":
            named_all_weights = [
                (name, param)
                for (name, param) in original_model.named_parameters()
                if param.dim() > 1
            ]
            attention_modules = []
            framework = None

        total_modules = len(named_all_weights) + len(attention_modules)

        # Phase 1: Scan MLP / Conv modules
        if named_all_weights:
            if verbose:
                print(f"Phase 1: Scanning {len(named_all_weights)} MLP/Conv modules...")
            layer_iter = tqdm(enumerate(named_all_weights), desc="MLP/Conv layers",
                              leave=False, total=len(named_all_weights))
            for i_layer, (name, _) in layer_iter:
                mod_name, best_sp = self._scan_single_module(
                    original_model, name, sparsities, dense_model_accuracy,
                    example_inputs, vit_config, verbose, i_layer, total_modules,
                    is_attention=False, framework=framework,
                )
                self.sparsity_dict[mod_name] = best_sp

        # Phase 2: Scan attention modules (head pruning via DG)
        if attention_modules:
            # For attention heads, use coarser sparsity steps based on head counts
            if verbose:
                print(f"Phase 2: Scanning {len(attention_modules)} attention modules (head pruning)...")
            attn_iter = tqdm(enumerate(attention_modules), desc="Attention heads",
                             leave=False, total=len(attention_modules))
            for i_attn, (name, _) in attn_iter:
                mod_name, best_sp = self._scan_single_module(
                    original_model, name, sparsities, dense_model_accuracy,
                    example_inputs, vit_config, verbose,
                    len(named_all_weights) + i_attn, total_modules,
                    is_attention=True, framework=framework,
                )
                self.sparsity_dict[mod_name] = best_sp

        # Restore original model and prune mode
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
        Supports Conv2d-based models (default) and ViT architectures (when attention_heads=True).

        For ViT models, uses a two-path approach:
        1. MLP intermediate layers: pruned via MetaPruner with per-module ratios
        2. Attention modules: pruned via DG-based head pruning (preserves embed_dim)
        """
        model_device = next(self.model.parameters()).device
        example_inputs = next(iter(self.dataloader['test']))[0][:1,].to(model_device)

        vit_config = self._detect_vit_config(self.model) if self.attention_heads else None

        if vit_config and vit_config['framework']:
            framework = vit_config['framework']
            prunable_modules = vit_config['prunable_modules']
            attention_modules = vit_config['attention_modules']

            # Split sparsity_dict into MLP ratios and attention ratios
            mlp_names = {name for name, _ in prunable_modules}
            attn_names = {name for name, _ in attention_modules}

            mlp_ratio_dict = {}
            attn_sparsities = {}
            for sname, sparsity in self.sparsity_dict.items():
                if sname in attn_names:
                    attn_sparsities[sname] = sparsity
                elif sname in mlp_names:
                    # Map name -> module object for MetaPruner
                    module = dict(self.model.named_modules())[sname]
                    mlp_ratio_dict[module] = sparsity

            # Step 1: Prune MLP layers via MetaPruner
            if mlp_ratio_dict:
                pruner_kwargs = {
                    'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
                    'pruning_ratio': 0,
                    'pruning_ratio_dict': mlp_ratio_dict,
                    'round_to': 8,
                }
                pruner_kwargs.update(self._build_metapruner_kwargs(vit_config))
                pruner = tp.pruner.MetaPruner(self.model, example_inputs, **pruner_kwargs)
                pruner.step()

            # Step 2: Prune attention heads via DG
            if attn_sparsities:
                if framework == 'timm':
                    self._patch_timm_attention(self.model)
                # Re-detect config after MLP pruning (module refs may have changed)
                current_vit_config = self._detect_vit_config(self.model)
                for name, sparsity in attn_sparsities.items():
                    if sparsity > 0:
                        attn_module = dict(self.model.named_modules())[name]
                        self._prune_attention_heads(
                            self.model, example_inputs, attn_module, name,
                            sparsity, framework, current_vit_config,
                        )
        else:
            # Original Conv2d-only path
            prunable_modules = [
                (name, module) for name, module in self.model.named_modules()
                if isinstance(module, nn.Conv2d)
            ]

            ratio_dict = {
                module: sparsity
                for (name, module), (sname, sparsity) in zip(prunable_modules, self.sparsity_dict.items())
            }

            pruner_kwargs = {
                'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
                'pruning_ratio': 0,
                'pruning_ratio_dict': ratio_dict,
                'round_to': 8,
            }

            pruner = tp.pruner.MetaPruner(self.model, example_inputs, **pruner_kwargs)
            pruner.step()
