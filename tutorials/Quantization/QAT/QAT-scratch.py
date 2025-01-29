import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import os
from tqdm import tqdm
import copy

# Configuration class
class Config:
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    EPOCHS = 5
    WEIGHT_BITS = 8
    ACTIVATION_BITS = 8
    PATIENCE = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuantizedLinear(nn.Module):
    def __init__(self, layer: nn.Linear, bit_width: int = 8, 
                 act_bit_width = 8, device: str = 'cpu'):
        super(QuantizedLinear, self).__init__()
            
        self.bit_width = bit_width
        self.act_bit_width = act_bit_width
        self.device = device

        self.is_conv = True if isinstance(layer, nn.Conv2d) else False
        if self.is_conv:
            self.stride = layer.stride
            self.padding = layer.padding
        
        self.weight = nn.Parameter(layer.weight.data.clone()).to(device)
        self.bias = nn.Parameter(layer.bias.data.clone()).to(device) if layer.bias is not None else None
        
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('qweight', None)
        
    def quantize_tensor(self, tensor: torch.Tensor, bit_width: int):
        tensor = tensor.float()
        max_val = tensor.abs().max()
        scale = max_val / (float(2 ** (bit_width - 1) - 1)) if max_val != 0 else 1.0
        q_tensor = torch.round(tensor / scale).clamp(
            min=-(2 ** (bit_width - 1)),
            max=(2 ** (bit_width - 1) - 1)
        )
        return q_tensor.to(torch.int8), scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qweight is None or self.training:
            with torch.no_grad():
                self.qweight, self.weight_scale = self.quantize_tensor(self.weight, self.bit_width)
                self.qweight = self.qweight.to(self.device)
        
        if self.is_conv:
            x = F.conv2d(x, self.qweight.float() * self.weight_scale, self.bias, stride=self.stride, padding=self.padding)
        else:
            x = F.linear(x, self.qweight.float() * self.weight_scale, self.bias)
        
        if self.act_bit_width is not None:
            x, self.act_scale = self.quantize_tensor(x, self.act_bit_width)
            x = x.float() * self.act_scale
        
        return x

    def __repr__(self):
        return f"QuantizedLinear(in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None})"

# Replace layers with quantized versions
def replace_layers(model: nn.Module, device: str):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quant_layer = QuantizedLinear(module, 
                                        bit_width=Config.WEIGHT_BITS,
                                        act_bit_width=Config.ACTIVATION_BITS,
                                        device=device)
            setattr(model, name, quant_layer)
        else:
            replace_layers(module, device)

# Data loaders
def get_data_loaders():
    image_size = 32
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms['train'], download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms['test'], download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# Model evaluation
def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Model training
def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer):
    for epoch in tqdm(range(Config.EPOCHS)):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model at the end of each epoch
        accuracy = evaluate_model(model, test_loader)
        

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
import copy


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    model = SimpleNN().to('cuda')
    print(f"\n Original Model: {model}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    train_model(model, train_loader, test_loader, criterion, optimizer)
    accuracy = evaluate_model(model, test_loader)
    print(f"Trained Model Accuracy: {accuracy:.2f}%")

    # Apply QAT by replacing layers
    qat_model = copy.deepcopy(model)
    replace_layers(qat_model, Config.DEVICE)
    print(f"Quantized Model: {qat_model}")
    qat_model.to(Config.DEVICE)  # Move the model to the specified device
    optimizer = optim.Adam(qat_model.parameters(), lr=Config.LEARNING_RATE)
    train_model(qat_model, train_loader, test_loader, criterion, optimizer)

    accuracy = evaluate_model(model, test_loader)
    print(f"Quantized Model Accuracy: {accuracy:.2f}%")
