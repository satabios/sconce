import torch

def pytorch_linear(input_data, kernel_weights, kernel_bias):
    return torch.matmul(input_data, kernel_weights.T)+ kernel_bias.unsqueeze(0)
    