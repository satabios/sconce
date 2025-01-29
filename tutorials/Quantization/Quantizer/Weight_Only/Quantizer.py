import torch
from Qop import Qop
import torch.nn as nn
import torch.nn.functional as F

class Quantizer(nn.Module):
    def __init__(
            self,
            in_features,  # C_in/Lin-in (CNN/Linear)
            out_features,  # C_out/Lin-out (CNN/Linear)
            kernel_size=None,
            stride=None,
            padding=None,
            dilation=None,
            groups=None,
            bias=None,
            quantize_output=False,
            cnn=False,
            data_metrics=None,
            activations=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cnn = cnn
        self.data_metrics = data_metrics
        self.kernel_size = kernel_size if cnn else None
        self.stride = stride if cnn else None
        self.padding = padding if cnn else None
        self.dilation = dilation if cnn else None
        self.groups = groups if cnn else None

        self.weight_shape = (self.out_features, self.in_features, *self.kernel_size) if cnn else (self.out_features, self.in_features)
        self.register_buffer("weight", torch.randn(self.weight_shape, dtype=torch.float16, requires_grad=False))
        self.bias = None if bias is None else bias

        self.input_quant = False
        self.input_quantizer = Qop(
            dtype=torch.uint8,
            symmetric=False,
            affine='tensor',
            affine_dim=None
        )
        self.input_quantizer.max_val = 127
        self.input_observer = torch.ao.quantization.observer.MinMaxObserver(dtype=torch.quint8,

                                                                            qscheme=torch.per_tensor_affine)#torch.ao.quantization.observer.MinMaxObserver(dtype=torch.int8) #

        self.weight_quant = Qop(
            dtype=data_metrics['weights']['dtype'],
            symmetric=data_metrics['weights']['symmetric'],
            affine=data_metrics['weights']['affine'],
            affine_dim=data_metrics['weights']['affine_dim']
        )

        self.quantize_output = quantize_output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self


    @torch.no_grad()
    def forward(self, x):

        # if self.input_quant: #Activated post to observer stats
        #     x = self.input_quantizer.quantize(x)

        if self.cnn:
            y = F.conv2d(x, self.weight.to(x.dtype), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            if self.bias is not None:
                # Ensure the bias tensor is reshaped for broadcasting
                bias_reshaped = self.bias.view(1, -1, 1, 1)  # For Conv2D
                y = y+ bias_reshaped
        else:
            y = F.linear(x, self.weight.to(x.dtype))
            if self.bias is not None:
                # Ensure the bias tensor is reshaped for broadcasting
                bias_reshaped = self.bias.view(1, -1)  # For Linear
                y = y+ bias_reshaped
        

        y  = self.weight_quant.dequantize(y, activation=True if self.weight_quant.affine=="channel" else False)

        # if self.input_quant:
        #     y = torch.round(y/self.input_quantizer.scales.to(y.device))# + self.input_quantizer.zero_point.to(y.device)).clamp(0,127)
            # tensor / self.scales + self.zero_point).clamp(self.q_min, self.q_max)
        return y

    @staticmethod
    def from_float(module, quantize_output=False, activations=None, data_metry=None):
        # print("Creating Quantizer from module:", module)
        new_module = Quantizer(
            in_features=module.in_features if isinstance(module, nn.Linear) else module.in_channels,
            out_features=module.out_features if isinstance(module, nn.Linear) else module.out_channels,
            kernel_size=module.kernel_size if isinstance(module, nn.Conv2d) else None,
            stride=module.stride if isinstance(module, nn.Conv2d) else None,
            padding=module.padding if isinstance(module, nn.Conv2d) else None,
            dilation=module.dilation if isinstance(module, nn.Conv2d) else None,
            groups=module.groups if isinstance(module, nn.Conv2d) else None,
            bias=module.bias,
            quantize_output=quantize_output,
            cnn=isinstance(module, nn.Conv2d),
            data_metrics=data_metry,
            activations=activations
        )
        new_module.weight = new_module.weight_quant.quantize(module.weight)
        new_module.input_observer = new_module.input_observer.to(new_module.weight.device)
        return new_module



    def __repr__(self):
        # Check if 'outputs' and 'weights' are not None and handle accordingly
        if self.data_metrics and 'outputs' in self.data_metrics and self.data_metrics['outputs']:
            sym = "sym" if self.data_metrics['outputs'].get('symentric', True) else "asym"
            output_details = f"outputq={sym}/{str(self.data_metrics['outputs'].get('dtype', '')).split('.')[-1]}/{self.data_metrics['outputs'].get('affine', '')}"
        else:
            output_details = "outputq=None"

        if self.data_metrics and 'weights' in self.data_metrics and self.data_metrics['weights']:
            sym = "sym" if self.data_metrics['weights'].get('symentric', True) else "asym"
            weight_details = f"weightq={sym}/{str(self.data_metrics['weights'].get('dtype', '')).split('.')[-1]}/{self.data_metrics['weights'].get('affine', '')}, {output_details}"
        else:
            weight_details = "weightq=None"

        if self.cnn:
            return f"QConv2d({self.in_features}, {self.out_features}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None} {weight_details})"
        else:
            return f"QLinear({self.in_features}, {self.out_features}, bias={self.bias is not None} {weight_details})"

