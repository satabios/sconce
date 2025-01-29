import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import numpy as np

from optimum.quanto import Calibration, QBytesTensor, qfloat8_e4m3fn, qfloat8_e5m2, qint4, qint8
from optimum.quanto.nn import QConv2d

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'
os.environ['MAX_JOBS'] = '12'

# Load CUDA kernel
conv2d_cuda = load(name='conv2d_int8', sources=['conv2d_w8a8.cu'])

# Define input parameters
batch_size = 32  # Use smaller batch size for easier debugging
channel_in = 3  # Use single channel for easier debugging
width = 512
height = 512
channel_out = 4  # Use single output channel for easier debugging
kernel_width = 3
kernel_height = 3
stride = 1
padding = 0

dtype = torch.int8
# Create random input and kernel
input_tensor = torch.randint(0, 7, (batch_size, channel_in, width, height), device='cuda', dtype=dtype)
kernel_tensor = torch.randint(0, 7, (channel_out, channel_in, kernel_width, kernel_height), device='cuda', dtype=dtype)

# Compute output dimensions
out_width = (width + 2 * padding - kernel_width) // stride + 1
out_height = (height + 2 * padding - kernel_height) // stride + 1

# Allocate output tensor
output_tensor = torch.zeros(batch_size, channel_out, out_height, out_width, device='cuda', dtype=torch.int32)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

num_iterations = 100
custom_conv_times = []
qcustom_conv_times = []
torch_conv_times = []




# conv2d_int8_implementation = torch.compile(conv2d_cuda.conv2d_int8, backend="inductor")

for _ in range(num_iterations):
    # Run CUDA kernel
    start_event.record()
    # conv2d_int8_implementation(input_tensor, kernel_tensor, output_tensor, stride, padding)
    conv2d_cuda.conv2d_int8(input_tensor, kernel_tensor, output_tensor, stride, padding)
    end_event.record()

    # Wait for the events to be recorded
    torch.cuda.synchronize()
    custom_conv_times.append(start_event.elapsed_time(end_event))


conv2d = torch.nn.Conv2d(channel_in, channel_out, kernel_size=3, bias=True).to(torch.float32).to('cuda')
qconv2d = QConv2d.from_module(conv2d, weights=qint8, activations=qint8)

for _ in range(num_iterations):
    # Run CUDA kernel
    start_event.record()
    with torch.no_grad(), Calibration():
        qout = qconv2d(input_tensor)
    end_event.record()

    # Wait for the events to be recorded
    torch.cuda.synchronize()
    qcustom_conv_times.append(start_event.elapsed_time(end_event))


# PyTorch's conv2d for comparison
input_tensor_pt = input_tensor.to(torch.float32)
kernel_tensor_pt = kernel_tensor.to(torch.float32)

for _ in range(num_iterations):
    start_event.record()
    output_tensor_pt = F.conv2d(input_tensor_pt, kernel_tensor_pt, stride=stride, padding=padding)
    end_event.record()

    # Wait for the events to be recorded
    torch.cuda.synchronize()
    torch_conv_times.append(start_event.elapsed_time(end_event))

# Calculate average times
average_custom_conv_time = np.mean(custom_conv_times)
average_torch_conv_time = np.mean(torch_conv_times)
average_qtorch_conv_time = np.mean(qcustom_conv_times)
# print(output_tensor, output_tensor_pt)
print('Outputs match:', torch.allclose(output_tensor.to(torch.float32), output_tensor_pt, atol=1e-2))
print(f"Average custom 2D convolution time over {num_iterations} iterations: {average_custom_conv_time:.3f} ms")
print(f"Average custom 2D qconvolution from quantom optimo time over {num_iterations} iterations: {average_qtorch_conv_time:.3f} ms")
print(f"Average PyTorch F.conv2d time over {num_iterations} iterations: {average_torch_conv_time:.3f} ms")
speed_up = (average_torch_conv_time-average_custom_conv_time)/average_custom_conv_time
print(f"Speed up compared to Hugginface-Quanto to Custom Implementation: {speed_up:.3f}")
torch.clear_autocast_cache()
torch.cuda.empty_cache()