import torch
import torch.nn as nn
import torch.fx as fx
from tabulate import tabulate  # For nice tabular display
import torch
import torch.nn as nn
from collections import defaultdict, deque
from typing import Type

import copy
import torch.nn.intrinsic as nni
import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
from collections import defaultdict
import copy
from collections import defaultdict, deque


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)   # Ensures same spatial size
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 5, padding=2)  # Adjusted for spatial compatibility
        self.conv5 = nn.Conv2d(256, 512, 7, padding=3)  # Adjusted
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 128, 1)  # 1x1 convolution, no padding needed
        self.conv9 = nn.Conv2d(512, 128, 1)
        self.conv10 = nn.Conv2d(256, 512, 6, padding=2)  # Matches spatial size after concat
        self.conv11 = nn.Conv2d(512, 1024, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        l1 = self.relu(self.conv4(x3))
        r1 = self.relu(self.conv5(x3))
        l2 = self.relu(self.conv6(l1))
        r2 = self.relu(self.conv7(r1))
        l3 = self.relu(self.conv8(l2))
        r3 = self.relu(self.conv9(r2))
        x = self.relu(self.conv10(torch.cat([l3, r3], dim=1)))
        x = self.relu(self.conv11(x))
        return x



class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256 * 7 * 7, 1000)
        self.test_relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.test_relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_layer_shapes(model, input_shape=(1, 3, 224, 224)):
    """Track shapes through the model layers."""
    shapes = []
    x = torch.randn(input_shape)
    hooks = []
    
    def hook_fn(module, input, output):
        input_shape = tuple(input[0].shape) if isinstance(input, (tuple, list)) else tuple(input.shape)
        output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else tuple(output[0].shape)
        shapes.append({
            'layer_name': type(module).__name__,
            'input_shape': input_shape,
            'output_shape': output_shape
        })
    
    # Register hooks for all layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.MaxPool2d, nn.ReLU)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return shapes



def tabulate_model(model, model_name):
    print(f"\n{model_name}:")
    original_shapes = get_layer_shapes(model)
    print(tabulate([[s['layer_name'], s['input_shape'], s['output_shape']] 
                    for s in original_shapes],
                    headers=['Layer', 'Input Shape', 'Output Shape'],
                    tablefmt='grid'))



def replacer(traced_model):
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            layer = traced_model.get_submodule(node.target)
            if isinstance(layer, nn.ReLU):
                setattr(traced_model, node.target, nn.LeakyReLU())
    traced_model.graph.lint()

    new_model = fx.GraphModule(traced_model, traced_model.graph)
    return new_model


def add_module(traced_model, target_module, new_module_factory):
    idx = 0
    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            layer = traced_model.get_submodule(node.target)
            if isinstance(layer, target_module):
                new_module = new_module_factory(layer.out_channels)
                module_name = f"BN{idx}"
                traced_model.add_submodule(module_name, new_module)
    
                with traced_model.graph.inserting_after(node):
                    new_node = traced_model.graph.call_module(module_name, args=(node,))
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node,)
                
                idx += 1
    
    new_model = fx.GraphModule(traced_model, traced_model.graph)

    return new_model


def dependency_grapher(model: torch.fx.GraphModule, layer_type: Type[nn.Module] = nn.Conv2d):

    # Initialize the dependency graph
    graph = defaultdict(list)
    all_layers = set()  # Track all layers of the specified type

    # Helper function to find all next layers of the specified type
    def find_next_layers(node):
        queue = deque(node.users)
        visited = set()
        next_layers = []
        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node.op == "call_module":
                layer = model.get_submodule(current_node.target)
                if isinstance(layer, layer_type):
                    next_layers.append(current_node)
                else:
                    # Skip non-relevant layers and continue traversal
                    queue.extend(current_node.users)
            else:
                # Skip non-module nodes (e.g., call_function, call_method) and continue traversal
                queue.extend(current_node.users)
        return next_layers

    # Iterate over the nodes in the FX-traced graph
    for node in model.graph.nodes:
        if node.op == "call_module":
            layer = model.get_submodule(node.target)
            if isinstance(layer, layer_type):
                all_layers.add(node.target)  # Add to the set of layers
                next_nodes = find_next_layers(node)
                for next_node in next_nodes:
                    graph[node.target].append(next_node.target)

    # # Add nodes with no dependencies to ensure they're accounted for
    # for layer in all_layers:
    #     if layer not in graph and all(layer not in deps for deps in graph.values()):
    #         graph[None].append(layer)
    grouped_dict = defaultdict(list)

    for key, value in graph.items():
        # Convert the value to a tuple if it's a list (to make it hashable for dictionary keys)
        if isinstance(value, list):
            value = tuple(value)
        grouped_dict[value].append(key)

    output_dict = {}
    for value, keys in grouped_dict.items():
        output_dict[tuple(keys)] = tuple(value)

    return output_dict


# Define fused layers for pruning
fused_layers = (
    nni.modules.fused.ConvReLU2d,
    nni.modules.fused.ConvBn2d,
    nni.modules.fused.ConvBnReLU2d,
)

# Function to calculate channel importance
def get_input_channel_importance(weight, dim= 1):
    in_channels = weight.shape[dim]
    importances = []
    for i_c in range(in_channels):
        if(dim == 1):
            channel_weight = weight.detach()[:, i_c]
        else:
            channel_weight = weight.detach()[i_c]
        importance = torch.norm(channel_weight)
        importances.append(importance.view(1))
    return torch.cat(importances)


def channel_prune(model, prune_ratio):


    pruned_model = copy.deepcopy(model)
    dependency_graph = dependency_grapher(pruned_model)
    # print("Model Dependency Graph")
    # for prev, next in dependency_graph.items():
    #     print(f"Dependency Graph for {prev} -> {next}")
    #     print(f"Dependency Layers for {[model.get_submodule(pre) for pre in prev]} -> {[model.get_submodule(pre) for pre in next]}")
    n_conv = len(dependency_graph)

    with torch.no_grad():
        for idx, (prev_convs, next_convs) in enumerate(dependency_graph.items()):
            prev_layers = [pruned_model.get_submodule(prev_node) for prev_node in prev_convs]
            next_layers = [pruned_model.get_submodule(next_node) for next_node in next_convs]

            channels = prev_layers[0].out_channels #Same as Input Channels of Next Layers
            n_keep = int(round(channels * (1. - prune_ratio)))

            prev_layer_importances = [torch.argsort(get_input_channel_importance(prev_layer.weight, dim=0), descending=True) for prev_layer in prev_layers]
            next_layer_importances = [torch.argsort(get_input_channel_importance(next_layer.weight), descending=True) for next_layer in next_layers]

            #Prune O/P Channels for Prev Layers
            concat_handler = len(prev_layers)
            for prev_layer, layer_importance in zip(prev_layers, prev_layer_importances):
                prev_layer.weight.copy_(torch.index_select(
                    prev_layer.weight.detach(), 0, layer_importance))
                prev_layer.weight.set_(prev_layer.weight.detach()[:int(n_keep/concat_handler)])
                if hasattr(prev_layer, "bias") and prev_layer.bias is not None: #Pick bias as well based on layer_importance
                    prev_layer.bias.set_(prev_layer.bias.detach()[:int(n_keep/concat_handler)])
                prev_layer.out_channels = n_keep
            #Prune I/O Channels for Next Layers
            for next_layer, layer_importance in zip(next_layers, next_layer_importances):
                next_layer.weight.copy_(torch.index_select(
                    next_layer.weight.detach(), 1, layer_importance))
                next_layer.weight.set_(next_layer.weight.detach()[:,:n_keep])

                next_layer.in_channels = n_keep
    pruned_model.recompile()
    return pruned_model