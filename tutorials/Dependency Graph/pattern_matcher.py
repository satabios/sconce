import torch
import torch.nn as nn
import torch.fx as fx
import copy
from typing import Type, Iterable, Dict, Any, Tuple, List
from torch.ao.nn.intrinsic import ConvReLU2d, LinearReLU 

def _parent_name(target: str) -> Tuple[str, str]:
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if not isinstance(modules[current_node.target], expected_type):
            return False
    return True

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def fuse_conv_bn_eval(conv: nn.Module, bn: nn.Module) -> nn.Module:
    assert not (conv.training or bn.training), "Fusion only works in eval mode."
    fused_conv = copy.deepcopy(conv)

    # Fuse the weights and biases
    fused_w, fused_b = fuse_conv_bn_weights(
        fused_conv.weight, fused_conv.bias,
        bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
    )

    # Wrap the fused weights and biases in nn.Parameter
    fused_conv.weight = nn.Parameter(fused_w)
    if fused_b is not None:
        fused_conv.bias = nn.Parameter(fused_b)
    else:
        fused_conv.bias = None

    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b, eps):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return conv_w, conv_b


def fuse_conv_relu_eval(conv: nn.Module, relu: nn.Module) -> nn.Module:
    assert not (conv.training or relu.training), "Fusion only works in eval mode."
    fused_conv = copy.deepcopy(conv)
    fused_conv.relu = relu
    return fused_conv


def fuse_linear_relu_eval(linear: nn.Module, relu: nn.Module) -> nn.Module:
    assert not (linear.training or relu.training), "Fusion only works in eval mode."
    fused_linear = copy.deepcopy(linear)
    fused_linear.relu = relu
    return fused_linear

def fuse(model: torch.nn.Module, inplace=False) -> torch.nn.Module:

    fused_model  = fuser(fuser(model))

    return fused_model

def fuser(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    patterns = [
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        # (nn.Conv2d, nn.ReLU),
        # (nn.Linear, nn.ReLU),
    ]

    if not inplace:
        model = copy.deepcopy(model)

    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    # Iterate over the graph and apply fusion
    for pattern in patterns:
        for node in list(fx_model.graph.nodes):  # Use list to avoid modifying the graph during iteration
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv/linear is used by other nodes
                    continue

                if pattern == (nn.Conv2d, nn.ReLU):
                    #Replace with ConvReLU2d
                    conv = modules[node.args[0].target]
                    relu = modules[node.target]
                    fused_conv = fuse_conv_relu_eval(conv, relu)
                    replace_node_module(node.args[0], modules, fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)

                elif pattern == (nn.Linear, nn.ReLU):
                    linear = modules[node.args[0].target]
                    relu = modules[node.target]
                    fused_linear = fuse_linear_relu_eval(linear, relu)
                    replace_node_module(node.args[0], modules, fused_linear)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)

                else:  # Handle Conv + BN fusion
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    replace_node_module(node.args[0], modules, fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)

    fx_model.recompile()
    return fx_model
