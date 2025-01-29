import numpy as np
import torch
import torch.nn.functional as F

#Custom Implementations
from softmax import pytorch_softmax
from maxpool2d import pytorch_maxpool2d
from linear import pytorch_linear
from convolution import pytorch_conv2d, numpy_conv2d


torch.manual_seed(4123)

if __name__ == "__main__":
    torcher = 1
    numpy = 0
    cuda = 0

    conv2d = 1
    linear = 1
    softmax = 1
    relu = 1
    maxpool2d = 1

    batch_size = 512
    if torcher:
        if(conv2d):
            # Add Dilation Feature
            input_data = torch.rand(batch_size, 3, 32, 32)
            kernel_weights = torch.rand(8, 3, 3, 3)

            padding = 0
            stride = 1

            torch_kernel = torch.nn.Conv2d(3, 8, 3, padding=padding, stride=stride)
            torch_kernel.weight.data = kernel_weights
            torch_out = torch_kernel(input_data)

            kernel_bias = torch_kernel.bias
            data_out = pytorch_conv2d(input_data, kernel_weights, kernel_bias, padding=padding, stride=stride)

            print("------------------ Conv2d ---------------------------")
            print("Custom PyTorch Conv Output Shape:", data_out.shape)
            print("Torch Built-in Conv Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))

        if(linear):
            input_data = torch.rand(batch_size, 256)
            kernel_weights = torch.rand(64, 256)

            input_dim, output_dim = 256, 64

            torch_kernel = torch.nn.Linear(256, 64)
            torch_kernel.weight.data = kernel_weights
            torch_out = torch_kernel(input_data)

            kernel_bias = torch_kernel.bias
            data_out = pytorch_linear(input_data, kernel_weights, kernel_bias)

            print("------------------ Linear ---------------------------")
            print("Custom PyTorch Function Output Shape:", data_out.shape)
            print("Torch Built-in Function Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))
        
        if(relu):
            input_data = torch.rand(batch_size, 256)

            torch_kernel = torch.nn.ReLU()
            torch_out = torch_kernel(input_data)

            data_out = torch.clip(input_data,min=0)

            print("------------------ ReLU ---------------------------")
            print("Custom PyTorch Function Output Shape:", data_out.shape)
            print("Torch Built-in Function Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))

        if(softmax):
            input_data = torch.rand(batch_size, 256)
            dim = -1
            torch_kernel = torch.nn.Softmax(dim=dim)
            torch_out = torch_kernel(input_data)

            data_out = pytorch_softmax(input_data, dim)
            
            print("------------------ Softmax ---------------------------")
            print("Custom PyTorch Function Output Shape:", data_out.shape)
            print("Torch Built-in Function Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))

        if(maxpool2d):

            input_data = torch.rand(batch_size,3,32,32)
            kernel_size = 3
            torch_kernel = torch.nn.MaxPool2d(kernel_size)
            torch_out = torch_kernel(input_data)

            data_out = pytorch_maxpool2d(input_data,kernel_size)
            print("------------------ MaxPool2d ---------------------------")
            print("Custom PyTorch Conv Output Shape:", data_out.shape)
            print("Torch Built-in Conv Output Shape:", torch_out.shape)
            print("Close Match:", torch.allclose(torch_out, data_out, atol=1e-4))



    if numpy:
        if(conv2d):
            input_data = np.random.rand(16, 3, 32, 32)
            kernel_weights = np.random.rand(8, 3, 3, 3)

            padding = 0
            stride = 1

            data_out = numpy_conv2d(input_data, kernel_weights, padding=padding, stride=stride)
            print("Numpy Conv Output Shape:", data_out.shape)
