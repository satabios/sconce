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
        self.bias = bias

        self.weight_qscale, self.weight_zero_point = None, None
        self.output_zero_point = None, None
        self.scalers = None

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self


    @torch.no_grad()
    def forward(self, x):


        if self.cnn:
            y = F.conv2d(x, self.weight.to(x.dtype), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            y = F.linear(x, self.weight.to(x.dtype))

        if self.bias is not None:
            y = y + self.bias.to(y.device)
        y = y * self.scalers.to(y.device) + self.output_zero_point.to(y.device)


        return y

    @staticmethod
    def from_float(module, quantize_output=False, activations=None, data_metry=None):

        if module.bias is not None:
            if(isinstance(module, nn.Conv2d)):
                bias_reshaped = module.bias.view(1, -1, 1, 1)
            elif(isinstance(module, nn.Linear)):
                bias_reshaped = module.bias.view(1, -1)  # For Linear
        new_module = Quantizer(
            in_features=module.in_features if isinstance(module, nn.Linear) else module.in_channels,
            out_features=module.out_features if isinstance(module, nn.Linear) else module.out_channels,
            kernel_size=module.kernel_size if isinstance(module, nn.Conv2d) else None,
            stride=module.stride if isinstance(module, nn.Conv2d) else None,
            padding=module.padding if isinstance(module, nn.Conv2d) else None,
            dilation=module.dilation if isinstance(module, nn.Conv2d) else None,
            groups=module.groups if isinstance(module, nn.Conv2d) else None,
            bias=bias_reshaped if module.bias is not None else torch.tensor(0),
            quantize_output=quantize_output,
            cnn=isinstance(module, nn.Conv2d),
            data_metrics=data_metry,
            activations=activations
        )

        weight_quant = Qop(
            dtype=data_metry['weights']['dtype'],
            symmetric=data_metry['weights']['symmetric'],
            affine=data_metry['weights']['affine'],
            affine_dim=data_metry['weights']['affine_dim']
        )


        new_module.weight = weight_quant.quantize(module.weight)
        new_module.weight_qscale, new_module.weight_zero_point = weight_quant.scales, weight_quant.zero_point

        return new_module



    def __repr__(self):
        # Check if 'outputs' and 'weights' are not None and handle accordingly
        if self.data_metrics and 'outputs' in self.data_metrics and self.data_metrics['outputs']:
            sym = "sym" if self.data_metrics['outputs'].get('symentric', True) else "asym"
           
        if self.data_metrics and 'weights' in self.data_metrics and self.data_metrics['weights']:
            sym = "sym" if self.data_metrics['weights'].get('symentric', True) else "asym"
            weight_details = f"weightq={sym}/{str(self.data_metrics['weights'].get('dtype', '')).split('.')[-1]}/{self.data_metrics['weights'].get('affine', '')}"
        else:
            weight_details = "weightq=None"

        if self.cnn:
            return f"QConv2d({self.in_features}, {self.out_features}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None} {weight_details})"
        else:
            return f"QLinear({self.in_features}, {self.out_features}, bias={self.bias is not None} {weight_details})"




class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return super().forward(x.float()).to(torch.int8)

class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based AvgPool
        return super().forward(x.float()).to(torch.int8)