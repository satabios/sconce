import torch
from ModelAnalyzer import ModelAnalyzer
from Quantizer import Quantizer
import torch.nn as nn
from Qop import Qop
from tqdm import tqdm

class Chunker(ModelAnalyzer):

    def __init__(self, model, calibiration_data):
        self.model = model
        self.calibiration_data = calibiration_data
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers
        self.interested_layers = []
        self.hooks = {}
        self.attached_hooks = []
        self.chunk()

    def replace_modules(self, module, target_class, look_out_for, module_name_to_exclude=""):

        def find_optimal_qdict(tensor, activation=False):
            qerror = float('inf')
            qdict = None
            for affine in [("channel",0), ("tensor",None)]:
                for symmetric in [False, True]:
                    test_qlayer = Qop(
                                        dtype=torch.int8,
                                        symmetric=symmetric,
                                        affine=affine[0],
                                        affine_dim=affine[1]
                                    )
                    quantized_tensor = test_qlayer.quantize(tensor)
                    dequantized_tensor = test_qlayer.dequantize(quantized_tensor, activation)
                    error = test_qlayer.compute_dequantization_error(quantized_tensor, dequantized_tensor)
                    if(error<qerror):
                        qerror = error
                        qdict =  {'dtype': torch.int8, 'symmetric': symmetric, 'affine': affine[0], 'affine_dim': affine[1]}
            return qdict

        import functools
        def _stat_observer(module, input, output, module_name):
            input = input[0]
            if(module_name not in self.hooks):
                self.hooks[module_name] = {'input_stats': [float('inf'), float('-inf')], 'output_stats':  [float('inf'), float('-inf')]}
            self.hooks[module_name]['type'] = type(module)
            self.hooks[module_name]['input_stats'] = [ min(self.hooks[module_name]['input_stats'][0], input.min()), max(self.hooks[module_name]['input_stats'][1], input.max()) ]
            self.hooks[module_name]['output_stats'] = [ min(self.hooks[module_name]['output_stats'][0], output.max()), max(self.hooks[module_name]['output_stats'][1], output.max()) ]

        def dim_changer(shape, tensor):
            dimension = list(shape).index(tensor.shape[0])
            new_dim = []
            for idx in range(len(shape)):
                if (idx == dimension):
                    new_dim.append(-1)
                else:
                    new_dim.append(1)
            return tensor.view(new_dim)

        for name, child in module.named_children():
            if isinstance(child, look_out_for) and not \
                    any([x == name for x in module_name_to_exclude]):

                if(target_class=='weights'):
                    affine = ("channel", 0)# if isinstance(child, torch.nn.Conv2d) else ("tensor", None)
                    qdict = {'dtype': torch.int8, 'symmetric': True, 'affine': affine[0], 'affine_dim': affine[1]}
                    # qdict = find_optimal_qdict(child.weight)
                    q_params = {'weights': qdict }
                    qlayer = Quantizer.from_float(module=child, data_metry=q_params, quantize_output=False)
                    setattr(module, name, qlayer)
                elif (target_class == 'attach_observers'):
                    self.attached_hooks.append(child.register_forward_hook(functools.partial(_stat_observer, module_name=name)))
                elif (target_class == 'misc_layers'):
                    from Quantizer import QuantizedAvgPool2d, QuantizedMaxPool2d
                    if(isinstance(child, torch.nn.MaxPool2d)):
                        setattr(module, name, QuantizedMaxPool2d(kernel_size=child.kernel_size, stride=child.stride, padding=child.padding))
                    elif(isinstance(child, torch.nn.AvgPool2d)):
                        setattr(module, name, QuantizedAvgPool2d(kernel_size=child.kernel_size, stride=child.stride, padding=child.padding))

                elif (target_class == 'compute_qstats'):

                    Qin =  Qop(
                        dtype=torch.int8,
                        symmetric=False,
                        affine='tensor',
                        affine_dim=None
                    )
                    Qout = Qop(
                        dtype=torch.int8,
                        symmetric=False,
                        affine='tensor',
                        affine_dim=None
                    )
                    Qin.min_val, Qin.max_val = self.hooks[name]['input_stats']
                    Qout.min_val, Qout.max_val = self.hooks[name]['output_stats']  # Replace with Activation Stats
                    child.input_qscale, child.input_zero_point = Qin.calculate_params()[0].item(), Qout.calculate_params()[0].item()
                    child.output_qscale, child.output_zero_point = Qout.calculate_params()[0].item(), Qout.calculate_params()[0]

                    if (child.weight.dim()==4):
                        qweight = child.weight.sum((1, 2, 3))
                        child.bias = (child.bias.squeeze() - qweight.to(torch.int32) * child.input_zero_point).squeeze().view(1,-1,1,1)
                        child.scalers = ((child.input_qscale * child.weight_qscale) / (child.output_qscale)).squeeze().view(1,-1,1,1)

                    elif (child.weight.dim()==2):
                        qweight = child.weight.sum(1)
                        child.bias = (child.bias.squeeze()  - qweight.to(torch.int32) * child.input_zero_point).squeeze().view(1,-1)
                        child.scalers = ((child.input_qscale * child.weight_qscale) / (child.output_qscale)).squeeze().view(1,-1)
                    # Formula is :
                    # y = Conv(x, w) + bias or y = Linear(x, w) + bias
                    # y = y * scalers + output_zero_point
            else:
                    # Recursively call the function for nested modules
                    self.replace_modules(child, target_class, look_out_for, module_name_to_exclude)

    def weight_quantize(self):
        self.replace_modules(module=self.model, target_class='weights', look_out_for = (torch.nn.Conv2d, torch.nn.Linear))

    def calibirate_model(self):
        print("\nCalibrating model...")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for input_data, _ in tqdm(self.calibiration_data):
                _ = self.model(input_data.to(device))
        print("Calibration done \n")

    def attach_observers_observe(self):
        self.replace_modules(module=self.model, target_class='attach_observers', look_out_for=(torch.nn.Conv2d, torch.nn.Linear))
        self.calibirate_model()
        #Remove Hooks
        for hook in self.attached_hooks:
            hook.remove()

    def compute_qstats(self):
        self.replace_modules(module=self.model, target_class='compute_qstats',look_out_for=(Quantizer))

    def misc_layers(self):
        self.replace_modules(module=self.model, target_class='misc_layers',look_out_for=(torch.nn.MaxPool2d, torch.nn.AvgPool2d))

    def chunk(self):
        self.attach_observers_observe()
        self.weight_quantize()
        self.compute_qstats()
        # self.misc_layers()

