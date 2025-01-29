import torch

def pytorch_maxpool2d(input_data, kernel_size, stride=0, padding=0):
    if padding > 0:
        input_data = torch.nn.functional.pad(input_data, (padding, padding, padding, padding))
    
    stride = kernel_size if stride==0 else stride
    batch_size, channels = input_data.shape[0], input_data.shape[1]
    height, width = input_data.shape[2], input_data.shape[3]
    kernel_height = (height-kernel_size+2*padding)//stride+1
    kernel_width = (width-kernel_size+2*padding)//stride+1
    output_data = torch.zeros(batch_size, channels, kernel_height, kernel_width)
    # Reoder the for loops depending on Row-Major or Columns-Major Memory Access
    for row in range(kernel_height):
        for col in range(kernel_width):
                output_data[:, :, row, col] = torch.amax(input_data[:, :, row*stride:row*stride+kernel_size, col*stride:col*stride+kernel_size], dim=(2, 3))
    
    return output_data