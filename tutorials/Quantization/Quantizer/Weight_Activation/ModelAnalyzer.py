import torch
import torch.nn as nn
import re
import numpy as np
from collections import OrderedDict

class ModelAnalyzer:
    """
    model_layer: This key holds the layers of the model as per their original order.
    Conv2d_BatchNorm2d_ReLU: Stores sequences of layers where a Conv2d layer is followed by BatchNorm2d and then ReLU activation.
    Conv2d_BatchNorm2d: Stores sequences of layers where a Conv2d layer is followed by BatchNorm2d.
    Linear_ReLU: Stores sequences of layers where a Linear layer is followed by ReLU activation.
    Linear_BatchNorm1d: Stores sequences of layers where a Linear layer is followed by BatchNorm1d.
    name_type_shape: This contains information about each layer, including its name, type, and shape.
    name_list: A list of names of all layers.
    type_list: A list of types of all layers.
    qat_layers: Stores layers that are candidates for Quantization-Aware Training (QAT). Specifically, layers from the keys Conv2d_BatchNorm2d_ReLU and Conv2d_BatchNorm2d are considered for QAT.
    model_summary: Summary information about the model, including layer types, input and output shapes, and parameters.
    catcher: This holds detailed information about layers, including their names, types, input data shapes (x), weights (w), and output data shapes (y).
    fusion_layers: Dictionary containing different types of fusion layers such as Conv2d_BatchNorm2d_ReLU, Conv2d_BatchNorm2d, etc., along with their respective layer sequences.
    sequences: Identified layer sequences based on predefined patterns like 'Conv2d_Linear', 'Linear_Linear', etc. Each sequence is associated with its matching layer names.
    """
    def __init__(self, model, calibiration_data=None):
        self.model = model
        self.mapped_layers = OrderedDict()
        self.calibiration_data = calibiration_data
        self.layer_mapping(self.model)


    def name_fixer(self, names):
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

    def get_all_layers(self, model, parent_name=''):
        layers = []
        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            test_name = "model." + full_name
            try:
                eval(test_name)
                layers.append((full_name, module))
            except:
                layers.append((self.reformat_layer_name(full_name), module))
            if isinstance(module, nn.Module):
                layers.extend(self.get_all_layers(module, parent_name=full_name))
        return layers

    def reformat_layer_name(self, str_data):
        def replace_pattern(match):
            return f'._modules[\'{match.group(1)}\']'
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

        # For torch.graphmodules, the layer names get prefixed with _modules with the index under a key, value of the module
        # isinstance(model, torch.fx.GraphModule) If graphmodule replace the layer_named with _modules
        # nn.Module: eval('model.features[0][0]')
        # FX Module: eval('model.features._modules[\'0\']._modules[\'0\']')
        if(isinstance(self.model, torch.fx.GraphModule)):
            str_data = re.sub(r'\[(\d+)\]', replace_pattern, str_data)
            str_data = re.sub(r'\.{2,}', '.', str_data)

        return str_data

    def summary_string_fixed(self, model, all_layers, data, model_name=None, batch_size=-1, dtypes=None):
        x, y = data
        input_size = x.size()

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
                    m_key = self.name_fixer([m_key])[0]

                summary[m_key] = OrderedDict()
                summary[m_key]["type"] = str(type(module)).split('.')[-1][:-2]
                summary[m_key]["x"] = input
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

                if isinstance(output, (list, tuple)):
                    summary[m_key]["y"] = [
                        [-1] + list(o)[1:] for o in output
                    ]
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["y"] = list(output.detach())

                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    summary[m_key]["w"] = module.weight
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                    summary[m_key]["weight_shape"] = module.weight.shape
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    summary[m_key]["b"] = module.bias
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        if isinstance(input_size, tuple):
            input_size = [input_size]

        model_device = next(iter(model.parameters())).device
        x = x.to(model_device)

        summary = OrderedDict()
        hooks = []

        for module_idx, (layer_name, module) in enumerate(all_layers):
            register_hook(module, module_idx)

        model(x)

        for h in hooks:
            h.remove()

        return summary




    def layer_mapping(self, model):
        all_layers = self.get_all_layers(model)
        x, y = next(iter(self.calibiration_data))
        model_summary = self.summary_string_fixed(model, all_layers, (x, y), model_name='model')

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
            fixed_name_list = self.name_fixer(r_name_list)
            name_type_shape[:, 0] = fixed_name_list

        layer_types = name_type_shape[:, 1]
        layer_shapes = name_type_shape[:, 2]

        def find_combinations_indices(layer_type, layer_name, combinations):
            # Sort combinations by length to prioritize longer combinations first
            combinations = sorted(combinations, key=len, reverse=True)
            results = {tuple(combo): [] for combo in
                       combinations}  # Initialize the dictionary to store indices for each combo
            used_indices = set()  # Set to keep track of indices that are already part of a combination

            # Loop through the list, checking for combinations
            for i in range(len(layer_type)):
                for combo in combinations:
                    if i + len(combo) <= len(layer_type) and all(
                            layer_type[i + j] == combo[j] for j in range(len(combo))):
                        # Check if any of the indices in the potential match have been used already
                        current_indices = set(range(i, i + len(combo)))
                        if not current_indices & used_indices:  # Ensure there is no overlap with already used indices
                            # Record the indices tuple if the combination matches and indices are free
                            results[tuple(combo)].append(list(layer_name[i + j] for j in range(len(combo))))
                            used_indices.update(current_indices)  # Mark these indices as used

            return results
      
        layer_name, layer_type = name_type_shape[:, 0], name_type_shape[:, 1]
        layer_name, layer_type = list(layer_name), list(layer_type)
        possible_combinations = [
            ["Conv2d", "BatchNorm2d"],
            ["Conv2d", "BatchNorm2d", "ReLU"],
            ["Conv2d", "ReLU"],
            ["Linear", "ReLU"],
            ["BatchNorm2d", "ReLU"],
            ['Conv2d'],
            ['Linear']

        ]
        mapped_layers = {'model_layer': []}
        mapped_layers['sequences'] = find_combinations_indices(layer_types, layer_name, possible_combinations)
        w_layers = [            ['Conv2d'],
            ['Linear']]
        mapped_layers['w_layers'] = find_combinations_indices(layer_types, layer_name, w_layers)
        fusable_layers = []
        for l_keys, fuse_layers in mapped_layers['sequences'].items():
            fusable_layers.extend(fuse_layers)
        mapped_layers['fusable_layers'] =  fusable_layers
        # Initialize containers for the current sequence of layers being analyzed and the final list of fusible layers







        mapped_layers['name_type_shape'] = name_type_shape
        mapped_layers['name_list'] = mapped_layers['name_type_shape'][:, 0]
        mapped_layers['type_list'] = mapped_layers['name_type_shape'][:, 1]

        mapped_layers['fusable_layers'] = fusable_layers
        mapped_layers['model_summary'] = model_summary

        name_list = mapped_layers['name_type_shape'][:, 0]
        layer_name_list, layer_type_list = [], []
        w, x, y, b = [], [], [], []
        calibiration_data = {}
        # for layer_name in name_list:
        #
        #     layer = eval(layer_name)
        #
        #     # isinstance(model, torch.fx.GraphModule) If graphmodule replace the layer_named with _modules
        #     # nn.Module: eval('model.features[0][0]')
        #     # FX Module: eval('model.features._modules[\'0\']._modules[\'0\']')
        #
        #     if (isinstance(layer, (nn.Conv2d, nn.Linear))):
        #
        #         layer_data =  {
        #                 'layer_name': layer_name,
        #                 'layer_type': (type(layer),str(type(layer)).split('.')[-1][:-2]),
        #                 'layer' : layer,
        #                 'activations': mapped_layers['model_summary'][layer_name]['x'][0],
        #                 'weights': mapped_layers['model_summary'][layer_name]['w'],
        #                 'outputs': torch.stack([y.contiguous().clone() for y in mapped_layers['model_summary'][layer_name]['y']])
        #         }
        #         calibiration_data.update({layer_name: layer_data})
        #
        # mapped_layers['calibiration_data'] = calibiration_data

        fusion_layers = ['Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d', 'Conv2d_ReLU', 'Linear_BatchNorm1d', 'Linear_ReLU']
        fusion_dict = {}
        for f_l in fusion_layers:
            if(f_l in mapped_layers.keys()):
                fusion_dict.update({f_l: mapped_layers[f_l]})
        mapped_layers['fusion_layers'] = fusion_dict


        self.mapped_layers = mapped_layers


        layers_to_remove  = ['model_layer', 'Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d', 'Linear_ReLU', 'Linear_BatchNorm1d',
                   'name_type_shape']
        for key in layers_to_remove:
            self.mapped_layers.pop(key, None)



