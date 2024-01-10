import torch
import torch.nn as nn
import re
import numpy as np
from torchsummary import summary
from collections import defaultdict, OrderedDict


def name_fixer(names):
    """
    Fix the names by removing the indices in square brackets.
    Args:
      names (list): List of names.

    Returns:
      list: List of fixed names.
    """
    return_list = []
    for string in names:
        matches = re.finditer(r'\.\[(\d+)\]', string)
        pop_list = [m.start(0) for m in matches]
        pop_list.sort(reverse=True)
        if len(pop_list) > 0:
            string = list(string)
            for pop_id in pop_list:
                string.pop(pop_id)
            string = ''.join(string)
        return_list.append(string)
    return return_list


class VGG(nn.Module):
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x


fusing_layers = [
    'Conv2d',
    'BatchNorm2d',
    'ReLU',
    'Linear',
    'BatchNorm1d',
]

import copy


def get_all_layers(model, parent_name=''):
    layers = []
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        test_name = "model." + full_name
        try:
            eval(test_name)
            layers.append((full_name, module))
        except:
            layers.append((reformat_layer_name(full_name), module))
        if isinstance(module, nn.Module):
            layers.extend(get_all_layers(module, parent_name=full_name))
    return layers


def reformat_layer_name(str_data):
    try:
        split_data = str_data.split('.')
        for ind in range(len(split_data)):
            data = split_data[ind]
            if (data.isdigit()):
                split_data[ind] = "[" + data + "]"
        final_string = '.'.join(split_data)

        iters_a = re.finditer(r'[a-zA-Z]\.\[', final_string)
        indices = [m.start(0) + 1 for m in iters_a]
        iters = re.finditer(r'\]\.\[', final_string)
        indices.extend([m.start(0) + 1 for m in iters])

        final_string = list(final_string)
        final_string = [final_string[i] for i in range(len(final_string)) if i not in indices]

        str_data = ''.join(final_string)

    except:
        pass

    return str_data


def summary_string_fixed(model, all_layers, input_size, model_name=None, batch_size=-1, dtypes=None):
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    def register_hook(module, module_idx):
        def hook(module, input, output):
            nonlocal module_idx
            m_key = all_layers[module_idx][0]
            m_key = model_name + "." + m_key

            try:
                eval(m_key)
            except:
                m_key = name_fixer([m_key])[0]

            summary[m_key] = OrderedDict()
            summary[m_key]["type"] = str(type(module)).split('.')[-1][:-2]
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["weight_shape"] = module.weight.shape
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [torch.rand(2, *in_size).type(dtype)
         for in_size, dtype in zip(input_size, dtypes)]

    summary = OrderedDict()
    hooks = []

    for module_idx, (layer_name, module) in enumerate(all_layers):
        register_hook(module, module_idx)

    model(*x)

    for h in hooks:
        h.remove()

    return summary


def get_input_channel_importance(weight):
    importances = []
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]
        importance = torch.norm(channel_weight)
        importances.append(importance.view(1))
    return torch.cat(importances)


def get_importance(layer, sparsity):
    sorted_indices = torch.argsort(get_input_channel_importance(layer.weight), descending=True)
    n_keep = int(round(len(sorted_indices) * (1.0 - sparsity)))
    indices_to_keep = sorted_indices[:n_keep]
    return indices_to_keep


def cwp_possible_layers(layer_name_list):
    possible_indices = []
    idx = 0
    while idx < len(layer_name_list):
        current_value = layer_name_list[idx]
        layer_shape = eval(current_value).weight.shape
        curr_merge_list = []
        curr_merge_list.append([current_value, 0])
        hit_catch = False
        for internal_idx in range(idx + 1, len(layer_name_list) - 1):
            new_layer = layer_name_list[internal_idx]
            new_layer_shape = eval(new_layer).weight.shape
            if len(new_layer_shape) == 4:
                curr_merge_list.append([new_layer, 0])
                if layer_shape[0] == new_layer_shape[1]:
                    hit_catch = True
                    break
            elif len(new_layer_shape) == 1:
                curr_merge_list[len(curr_merge_list) - 1][1] = new_layer
        possible_indices.append(curr_merge_list)
        if hit_catch == True:
            idx = internal_idx
        else:
            idx += 1
    return possible_indices


import copy


@torch.no_grad()
def prune_cwp(model):
    pruned_model = copy.deepcopy(model)
    pruning_ratio = 0.10
    pruning_layer_length = len(possible_indices_ranges)
    pruning_ratio_list = (pruning_layer_length) * [pruning_ratio]

    def get_layer_name(obj):
        if isinstance(obj, list):
            layer_list = []
            for internal_layer in obj:
                layer_list.append(eval(internal_layer.replace('model', 'pruned_model')))
            return layer_list
        else:
            nonlocal pruned_model
            return eval(obj.replace('model', 'pruned_model'))

    for list_ind in range(len(possible_indices_ranges)):
        sparsity = pruning_ratio_list[list_ind]
        layer_list = np.asarray(possible_indices_ranges[list_ind])
        prev_conv = get_layer_name(layer_list[0, 0])
        prev_bn = get_layer_name(layer_list[0, 1])
        next_convs = [c for c in get_layer_name(list(layer_list[1:, 0]))]
        next_bns = [b for b in get_layer_name(list(layer_list[1:-1, 1]))]  # Avoid last 0

        if (len(next_bns) == 0):
            iter_layers = zip([prev_conv, prev_bn], [next_convs, []])
        else:
            iter_layers = zip([prev_conv, prev_bn], [next_convs, next_bns])

        importance_list_indices = get_importance(layer=next_convs[-1], sparsity=sparsity)

        def prune_bn(layer, importance_list_indices):

            layer.weight.set_(layer.weight.detach()[importance_list_indices])
            layer.bias.set_(layer.bias.detach()[importance_list_indices])
            layer.running_mean.set_(layer.running_mean.detach()[importance_list_indices])
            layer.running_var.set_(layer.running_var.detach()[importance_list_indices])

        for prev_layer, next_layers in iter_layers:
            print("Prev:", prev_layer.weight.shape)
            print("Next:", [next.weight.shape for next in next_layers])

            if (str(type(prev_layer)).split('.')[-1][:-2] == 'BatchNorm2d'):  # BatchNorm2d
                prune_bn(prev_layer, importance_list_indices)
            else:
                prev_layer.weight.set_(prev_conv.weight.detach()[importance_list_indices, :])

            if (len(next_layers) != 0):
                for next_layer in next_layers:
                    if (str(type(next_layer)).split('.')[-1][:-2] == 'BatchNorm2d'):  # BatchNorm2d
                        prune_bn(next_layer, importance_list_indices)
                    else:
                        if (next_layer.weight.shape[1] == 1):

                            next_layer.weight.set_(next_layer.weight.detach()[importance_list_indices, :])
                            next_layer.groups = len(importance_list_indices)

                        else:

                            next_layer.weight.set_(next_layer.weight.detach()[:, importance_list_indices])

    return pruned_model, model


def layer_mapping(model):
    all_layers = get_all_layers(model)
    model_summary = summary_string_fixed(model, all_layers, (3, 64, 64), model_name='model')  # , device="cuda")

    name_type_shape = []
    for key in model_summary.keys():
        data = model_summary[key]
        if ("weight_shape" in data.keys()):
            name_type_shape.append([key, data['type'], data['weight_shape'][0]])
        #     else:
    #         name_type_shape.append([key, data['type'], 0 ])
    name_type_shape = np.asarray(name_type_shape)

    name_list = name_type_shape[:, 0]

    r_name_list = np.asarray(name_list)
    random_picks = np.random.randint(0, len(r_name_list), 10)
    test_name_list = r_name_list[random_picks]
    eval_hit = False
    for layer in test_name_list:
        try:
            eval(layer)

        except:
            eval_hit = True
            break
    if (eval_hit):
        fixed_name_list = name_fixer(r_name_list)
        name_type_shape[:, 0] = fixed_name_list

    layer_types = name_type_shape[:, 1]
    layer_shapes = name_type_shape[:, 2]
    mapped_layers = {'model_layer': [], 'Conv2d_BatchNorm2d_ReLU': [], 'Conv2d_BatchNorm2d': [], 'Linear_ReLU': [],
                     'Linear_BatchNorm1d': []}

    def detect_sequences(lst):
        i = 0
        while i < len(lst):

            if i + 2 < len(lst) and [l for l in lst[i: i + 3]] == [
                fusing_layers[0],
                fusing_layers[1],
                fusing_layers[2],
            ]:
                test_layer = layer_shapes[i: i + 2]
                if (np.all(test_layer == test_layer[0])):
                    mapped_layers['Conv2d_BatchNorm2d_ReLU'].append(
                        np.take(name_list, [i for i in range(i, i + 3)]).tolist()
                    )
                    i += 3

            elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                fusing_layers[0],
                fusing_layers[1],
            ]:
                test_layer = layer_shapes[i: i + 2]
                if (np.all(test_layer == test_layer[0])):
                    mapped_layers['Conv2d_BatchNorm2d'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
            # if i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[0], fusing_layers[2]]:
            #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
            #     i += 2
            # elif i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[1], fusing_layers[2]]:
            #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
            #     i += 2
            elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                fusing_layers[3],
                fusing_layers[2],
            ]:
                mapped_layers['Linear_ReLU'].append(
                    np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                )
                i += 2
            elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                fusing_layers[3],
                fusing_layers[4],
            ]:
                mapped_layers['Linear_BatchNorm1d'].append(
                    np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                )
                i += 2
            else:
                i += 1

    detect_sequences(layer_types)

    for keys, value in mapped_layers.items():
        mapped_layers[keys] = np.asarray(mapped_layers[keys])

    mapped_layers['name_type_shape'] = name_type_shape
    # self.mapped_layers = mapped_layers

    # CWP
    keys_to_lookout = ['Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d']
    pruning_layer_of_interest, qat_layer_of_interest = [], []

    # CWP or QAT Fusion Layers
    for keys in keys_to_lookout:
        data = mapped_layers[keys]
        if (len(data) != 0):
            qat_layer_of_interest.append(data)
    mapped_layers['qat_layers'] = np.asarray(qat_layer_of_interest)

    return mapped_layers


# GMP
#         layer_of_interest=mapped_layers['name_type_shape'][:,0] # all layers with weights
#         Check for all with weights
# Wanda

def string_fixer(name_list):
    for ind in range(len(name_list)):
        modified_string = re.sub(r'\.(\[)', r'\1', name_list[ind])
        name_list[ind] = modified_string


# #load the pretrained model

# model = timm.create_model('mobilenetv2_100', num_classes=10)
# model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

model = VGG()
mapped_layers = layer_mapping(model)
name_list = mapped_layers['name_type_shape'][:, 0]

possible_indices_ranges = cwp_possible_layers(name_list)
possible_indices_ranges = [lst for lst in possible_indices_ranges if len(lst) > 1]
possible_indices_ranges = possible_indices_ranges[:-1]

pruned_model, original_model = prune_cwp(model)

summary(pruned_model, (3, 32, 32))

# summary_string_fixed(pruned_model, (3, 64, 64),model_name ='pruned_model')
