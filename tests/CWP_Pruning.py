
import torch
import torch.nn as nn
from collections import OrderedDict
import re
import numpy as np
from torchsummary import summary
from collections import defaultdict, OrderedDict
import ipdb

from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import torch.optim as optim



image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    "test": ToTensor(),
}
dataset = {}
for split in ["train", "test"]:
  dataset[split] = CIFAR10(
    root="data/cifar10",
    train=(split == "train"),
    download=True,
    transform=transforms[split],
  )
dataloader = {}
for split in ['train', 'test']:
  dataloader[split] = DataLoader(
    dataset[split],
    batch_size=512,
    shuffle=(split == 'train'),
    num_workers=0,
    pin_memory=True,
  )


from sconce import sconce

sconces = sconce()
sconces.criterion = nn.CrossEntropyLoss() # Loss

sconces.dataloader = dataloader
sconces.epochs = 5 #Number of time we iterate over the data
sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sconces.experiment_name = "vgg-gmp" # Define your experiment name here
sconces.prune_mode = "GMP"

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
    model_device = next(model.parameters()).device
    x = [torch.rand(2, *in_size).type(dtype).to(model_device)
         for in_size, dtype in zip(input_size, dtypes)]

    summary = OrderedDict()
    hooks = []

    for module_idx, (layer_name, module) in enumerate(all_layers):
        register_hook(module, module_idx)

    model(*x)

    for h in hooks:
        h.remove()

    return summary



def layer_mapping(model):

    def cwp_layer_detections(name_list):
        sorting_pairs = []
        idx = 0
        while idx < len(name_list):

            layer_list = [ layer_name for layer_name in name_list[idx:idx+3]]

            if([ type(eval(l)) for l in layer_list] == [nn.Conv2d, nn.BatchNorm2d,nn.Conv2d]):
                sorting_pairs.append(layer_list)
                idx+=2
            elif([ type(eval(l)) for l in layer_list[idx:idx+2]] == [nn.Conv2d, nn.Conv2d]):
                sorting_pairs.append(layer_list[idx:idx+2].insert(1,0))
                idx+=1
            else:
                idx+=1
        return sorting_pairs
    def detect_sequences(lst):

        i = 0
        while i < len(lst):

            if i + 2 < len(lst) and [l for l in lst[i: i + 3]] == [
                fusing_layers[0],
                fusing_layers[1],
                fusing_layers[2],
            ]:

                #                 test_layer = layer_shapes[i: i + 2]
                #                 if (np.all(test_layer == test_layer[0])):
                mapped_layers['Conv2d_BatchNorm2d_ReLU'].append(
                    np.take(name_list, [i for i in range(i, i + 3)]).tolist()
                )
                i += 3

            elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                fusing_layers[0],
                fusing_layers[1],
            ]:
                #                 test_layer = layer_shapes[i: i + 2]
                #                 if (np.all(test_layer == test_layer[0])):
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

    # model = copy.deepcopy(original_model)
    all_layers = get_all_layers(model)
    model_summary = summary_string_fixed(model, all_layers, (3, 64, 64), model_name='model')  # , device="cuda")

    name_type_shape = []
    for key in model_summary.keys():
        data = model_summary[key]
        if ("weight_shape" in data.keys()):
            name_type_shape.append([key, data['type'], data['weight_shape'][0]])
        else:
            name_type_shape.append([key, data['type'], 0])
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

    mapped_layers['qat_layers'] = qat_layer_of_interest
    name_list = mapped_layers['name_type_shape'][:, 0]
    type_list = mapped_layers['name_type_shape'][:, 1]
    conv_bn_list = name_list[[index for index, layer in enumerate(type_list) if layer in ['Conv2d', 'BatchNorm2d']]]


    mapped_layers['conv_bn_list'] = cwp_layer_detections(conv_bn_list)

    return mapped_layers


def string_fixer(name_list):
    for ind in range(len(name_list)):
        modified_string = re.sub(r'\.(\[)', r'\1', name_list[ind])
        name_list[ind] = modified_string


def get_input_channel_importance_original(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]

        importance = torch.norm(channel_weight)

        importances.append(importance.view(1))
    return torch.cat(importances)

@torch.no_grad()
def apply_channel_sorting_original(model, layer_list):
    model = copy.deepcopy(model)
    for i_conv in range(len(layer_list) - 1):

        prev_conv = eval(layer_list[i_conv][0])#.replace('model','sorted_model'))
        prev_bn = eval(layer_list[i_conv][1])#.replace('model','sorted_model'))
        next_conv = eval(layer_list[i_conv][2])#.replace('model','sorted_model'))
        # note that we always compute the importance according to input channels
        print(prev_conv.weight.shape, next_conv.weight.shape)
        # if((prev_conv.weight.shape[1]!=1) and (next_conv.weight.shape[1]!=1)):
        importance = get_input_channel_importance_original(next_conv.weight)
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

        next_conv.weight.copy_(
            torch.index_select(next_conv.weight.detach(), 1, sort_idx))

    return model

# vgg = VGG()
# checkpoint = torch.load("/home/sathya/Desktop/test-bed/vgg.cifar.pretrained.pth",map_location='cpu')
# vgg.load_state_dict(checkpoint["state_dict"])

vgg = VGG()
checkpoint = torch.load("/home/sathya/Desktop/test-bed/vgg.cifar.pretrained.pth",map_location='cpu')
vgg.load_state_dict(checkpoint["state_dict"])


mobilenet_v2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
mobilenet_v2.load_state_dict(torch.load("/home/sathya/Desktop/test-bed/mobilenet_v2-cifar10.pth"))


model = copy.deepcopy(mobilenet_v2)
mapped_layers = layer_mapping(model)

model_to_be_sorted = copy.deepcopy(mobilenet_v2)
sorted_model = apply_channel_sorting_original(model_to_be_sorted, mapped_layers['conv_bn_list'])



sconces.model= sorted_model # Model Definition
sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)

sconces.evaluate(verbose=True)