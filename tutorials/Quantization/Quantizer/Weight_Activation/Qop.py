import torch
import torch.nn as nn
import torch.nn.functional as F

class Qop:
    """Quantization operation supporting int8 and bfloat16 data types"""

    def __init__(self, dtype=None,bits=None, w_a=None, affine='tensor', affine_dim=None, group_size=-1, symmetric=False):
        self.symmetric = symmetric
        if(dtype is not None):
            self.dtype = dtype
            info = torch.finfo if dtype.is_floating_point else torch.iinfo
            self.q_min = info(self.dtype).min
            self.q_max = info(self.dtype).max
        

        #Hard Coded for Now
        # tensor_dtype = torch.float32
        # tensor_info = torch.finfo if tensor_dtype.is_floating_point else torch.iinfo
        self.eps = torch.tensor(torch.finfo(torch.float32).eps).detach()

        # self.eps = None
        self.min_val = None
        self.max_val = None
        self.w_a = w_a
        self.q_group_size = group_size

        # Defines the granularity of quantization: per tensor, channel, or group
        self.affine = affine
        self.affine_dim = affine_dim

        self.scales = None
        self.zero_point = None

    @torch.no_grad()
    def calculate_params(self):
        # Use torch.tensor(0) to ensure same type and device
        min_val_neg = torch.min(self.min_val, torch.tensor(0, device=self.min_val.device, dtype=self.min_val.dtype))
        max_val_pos = torch.max(self.max_val, torch.tensor(0, device=self.max_val.device, dtype=self.max_val.dtype))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if self.symmetric:
            # Symmetric Channel/Tensor-Wise
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(self.q_max - self.q_min) / 2)
            scale = torch.max(scale, self.eps)

            if self.dtype in [torch.quint8, torch.uint8]:
                zero_point = zero_point.new_full(zero_point.size(), (self.q_min + self.q_max) // 2)
        elif not self.symmetric and self.affine == "channel" and self.affine_dim is not None:
            # Affine Channel-Wise
            scale = (max_val_pos - min_val_neg) / float(self.q_max - self.q_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            zero_point = (-1 * min_val_neg / scale).to(torch.int)
        else:
            # Affine Tensor-Wise
            scale = (max_val_pos - min_val_neg) / float(self.q_max - self.q_min)
            scale = torch.max(scale, self.eps)
            zero_point = (self.q_min - torch.round(min_val_neg / scale)).to(torch.int)
            zero_point = torch.clamp(zero_point, self.q_min, self.q_max)

        if len(scale.shape) == 0:
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)

        return scale, zero_point

    @torch.no_grad()
    def compute_scales_zero_point(self, tensor=None):
        """Computes scale and zero-point based on tensor's min/max values."""
        if self.min_val is None and self.max_val is None:
            if self.affine=="channel":
                min_val, max_val = [], []
                dim_out = tensor.shape[self.affine_dim]
                for idx in range(dim_out):
                    tens=tensor[idx]
                    if self.symmetric:
                        max_val.append(torch.amax(tens))
                    else:
                        min_val.append(torch.max(tens))
                        max_val.append(torch.amax(tens))

                self.max_val = torch.tensor(max_val)
                if self.symmetric:
                    self.min_val = torch.zeros_like(self.max_val)
                # self.min_val = torch.tensor(self.min_val).detach()
            else:
                if self.symmetric:
                    self.max_val = tensor.abs().max()
                    self.min_val = torch.tensor(0, device=tensor.device)
                else:
                    self.max_val = tensor.max()
                    self.min_val = tensor.min()

        scales, zero_point = self.calculate_params()
        return scales, zero_point

    @torch.no_grad()
    def compute_scales_zero_point_dimension(self, tensor, dim=-1):

        self.scales, self.zero_point = self.compute_scales_zero_point(tensor)
        scale_shape = [1] * tensor.dim()

        output_dim = tensor.shape[dim]
        scale_shape[dim] = output_dim
        self.scales = self.scales.view(scale_shape)
        self.zero_point = self.zero_point.view(scale_shape)

    def compute_scale_zero_pointer(self):
        """Computes the scale and zero-point based on the affine type (tensor, channel, or group)."""
        if self.affine == 'tensor' or self.affine == 'channel':  # Per Tensor
            self.compute_scales_zero_point_dimension(self.tensor, dim=self.affine_dim)

        elif self.affine == 'group':  # Per Group (only for Linear layers)
            assert self.tensor.shape[1] % self.q_group_size == 0
            assert self.tensor.dim() == 2  # Only for Linear layers

            tensor = self.tensor.view(-1, self.q_group_size)
            self.compute_scales_zero_point_dimension(tensor, dim=0)

    def push_to_tensor_device(self, tensor_device):
        """Moves scale and zero-point to the specified device."""
        if isinstance(self.zero_point, torch.Tensor):
            self.zero_point = self.zero_point.clone().detach().to(tensor_device)
        else:
            self.zero_point = torch.tensor(self.zero_point).to(tensor_device)

        if isinstance(self.scales, torch.Tensor):
            self.scales = self.scales.clone().detach().to(tensor_device)
        else:
            self.scales = torch.tensor(self.scales).to(tensor_device)

    def quantize(self, tensor):
        """Quantizes the input tensor."""
        self.tensor = tensor
        self.tensor_shape = self.tensor.shape

        if self.zero_point is None or self.scales is None:  # If not pre-computed, calculate them
            self.compute_scale_zero_pointer()

        tensor = self.tensor.detach().clone()
        self.push_to_tensor_device(tensor.device)

        if self.affine == 'group':  # Handle group quantization (for Linear layers)
            orig_tensor_shape = tensor.shape
            tensor = tensor.view(tensor.shape[0] * (tensor.shape[1] // self.q_group_size), -1)

        if self.symmetric:
            self.quantized_tensor = torch.round(tensor / self.scales).clamp(self.q_min, self.q_max)
        else:
            self.quantized_tensor = torch.round(tensor / self.scales + self.zero_point).clamp(self.q_min, self.q_max)

        if self.affine == 'group':  # Reshape back after quantization
            self.quantized_tensor = self.quantized_tensor.view(orig_tensor_shape)

        return self.quantized_tensor.type(self.dtype)
    

    @torch.no_grad()
    def dequantize(self, quantized_tensor, activation=False):
        """Dequantizes the quantized tensor."""
        self.push_to_tensor_device(quantized_tensor.device)

        if self.affine == 'group':  # Handle group dequantization (for Linear layers)
            reshaped_tensor = quantized_tensor.view(
                quantized_tensor.shape[0] * (quantized_tensor.shape[1] // self.q_group_size), -1)
            dequantized_tensor = self.scales * (reshaped_tensor.float() - self.zero_point)
            self.dequantized_tensor = dequantized_tensor.view(quantized_tensor.shape)
        else:
            if activation and self.scales.shape[1] == 1:
                self.zero_point = self.zero_point.view([1, self.zero_point.shape[0],*self.zero_point.shape[2:] ])
                self.scales = self.scales.view([1, self.scales.shape[0],*self.scales.shape[2:] ])

            self.dequantized_tensor = self.scales * (quantized_tensor.float() - self.zero_point)

        return self.dequantized_tensor

    @torch.no_grad()
    def compute_dequantization_error(self, original_tensor, dequantized_tensor):
        """Computes the mean squared error between original and dequantized tensors."""
        if torch.isinf(original_tensor).any() or torch.isinf(dequantized_tensor).any():
            print("Inf values detected")
        if torch.isnan(original_tensor).any() or torch.isnan(dequantized_tensor).any():
            print("NaN values detected")

        # Normalize the tensors to avoid scale issues
        max_value = dequantized_tensor.abs().max()
        if max_value > 0:
            dequantized_tensor /= max_value

        return F.mse_loss(original_tensor, dequantized_tensor)
