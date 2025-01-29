import numpy as np
import torch


def numpy_conv2d(input_data, kernel_weights, padding=0, stride=1):
    if padding > 0:
        input_data = np.pad(input_data, 
                            ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                            mode='constant', constant_values=0)
    
    batch_size = input_data.shape[0]
    input_channels, output_channels = kernel_weights.shape[1], kernel_weights.shape[0]
    kernel_height, kernel_width = kernel_weights.shape[2], kernel_weights.shape[3]
    input_height, input_width = input_data.shape[2], input_data.shape[3]

    output_height = int((input_height - kernel_height) / stride) + 1
    output_width = int((input_width - kernel_width) / stride) + 1

    output_data = np.zeros((batch_size, output_channels, output_height, output_width))

    for output_channel in range(output_channels):
        kernel_data = kernel_weights[output_channel, :, :, :]
        for height in range(output_height):
            for width in range(output_width):
                region = input_data[:, :, height * stride:height * stride + kernel_height,
                                    width * stride:width * stride + kernel_width]
                output_data[:, output_channel, height, width] = np.sum(
                    region * kernel_data, axis=(1, 2, 3)
                )
    return output_data

def pytorch_conv2d(input_data, kernel_weights, kernel_bias, padding=0, stride=1):
    if padding > 0:
        input_data = F.pad(input_data, (padding, padding, padding, padding), "constant", 0)

    batch_size = input_data.shape[0]
    input_channels, output_channels = kernel_weights.shape[1], kernel_weights.shape[0]
    kernel_height, kernel_width = kernel_weights.shape[2], kernel_weights.shape[3]
    input_height, input_width = input_data.shape[2], input_data.shape[3]

    output_height = int((input_height - kernel_height) / stride) + 1
    output_width = int((input_width - kernel_width) / stride) + 1

    output_data = torch.zeros((batch_size, output_channels, output_height, output_width))

    for out_channel in range(output_channels):
        kernel_data = kernel_weights[out_channel, :, :, :]
        for height in range(output_height):
            for width in range(output_width):
                region = input_data[:, :, height * stride:height * stride + kernel_height,
                                    width * stride:width * stride + kernel_width]
                output_data[:, out_channel, height, width] = torch.sum(
                    region * kernel_data, dim=(1, 2, 3)
                )
            
    return output_data + kernel_bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # Expanding along Batch, Height and Width


import torch
import torch.nn.functional as F

# Define kernel as a tensor
kernel = torch.tensor([
    [[0., 1., 2.],
     [1., 2., 3.],
     [2., 3., 4.]],

    [[1., 2., 3.],
     [2., 3., 4.],
     [3., 4., 5.]]
]).unsqueeze(1)  # shape (2, 1, 3, 3) -> 2 output channels, 1 input channel, 3x3 kernel size

# Define feature map as a tensor
feature_map = torch.tensor([
    [0., 1., 2., 3., 4.],
    [1., 2., 3., 4., 5.],
    [2., 3., 4., 5., 6.],
    [3., 4., 5., 6., 7.],
    [4., 5., 6., 7., 8.]
]).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 5, 5) -> batch size 1, 1 channel, 5x5 feature map size

# Apply padding=1 and stride=1
output = F.conv2d(feature_map, kernel, padding=1, stride=1)

# Display output
print("Output Feature Map:")
for i, out_channel in enumerate(output[0]):
    print(f"Channel {i + 1}:")
    print(out_channel)
    print()
