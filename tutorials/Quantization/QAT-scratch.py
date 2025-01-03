import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import logging
from torchvision.datasets import *
from torchvision.transforms import *
from collections import OrderedDict, defaultdict
from typing import Optional, Tuple
import os
from datetime import datetime
import copy

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
                 act_bit_width: Optional[int] = 8, device: str = 'cpu'):
        super(QuantizedLinear, self).__init__()
        if bit_width <= 0 or (act_bit_width is not None and act_bit_width <= 0):
            raise ValueError("Bit width must be positive")
            
        self.bit_width = bit_width
        self.act_bit_width = act_bit_width
        self.device = device

        self.is_conv = True if isinstance(layer, nn.Conv2d) else False
        if self.is_conv:
            self.stride = layer.stride
            self.padding = layer.padding
        
        # Register parameters
        self.weight = nn.Parameter(layer.weight.data.clone()).to(device)
        self.bias = nn.Parameter(layer.bias.data.clone()).to(device) if layer.bias is not None else None
        
        # Register buffers for quantization weights, scales
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('qweight', None)
        
    def quantize_tensor(self, tensor: torch.Tensor, bit_width: int) -> Tuple[torch.Tensor, float]:
        """Quantize a tensor to the specified bit width."""
        tensor = tensor.float()
        max_val = tensor.abs().max()
        scale = max_val / (float(2 ** (bit_width - 1) - 1)) if max_val != 0 else 1.0
        q_tensor = torch.round(tensor / scale).clamp(
            min=-(2 ** (bit_width - 1)),
            max=(2 ** (bit_width - 1) - 1)
        )
        return q_tensor.to(torch.int8), scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights (only if needed)
        if self.qweight is None or self.training:
            with torch.no_grad():
                self.qweight, self.weight_scale = self.quantize_tensor(self.weight, self.bit_width)
                self.qweight = self.qweight.to(self.device)
        
        # Perform linear operation with quantized weights
        if self.is_conv:
            x = F.conv2d(x, self.qweight.float() * self.weight_scale, self.bias, stride=self.stride, padding=self.padding)
        else:
            x = F.linear(x, self.qweight.float() * self.weight_scale, self.bias)
        
        # Quantize activations if specified
        if self.act_bit_width is not None:
            x, self.act_scale = self.quantize_tensor(x, self.act_bit_width)
            x = x.float() * self.act_scale
        
        return x

def get_data_loaders():
    """Create and return data loaders."""
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
def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate model accuracy."""
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
    
    accuracy = 100 * correct / total
    return accuracy

def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[object] = None):
    """Train the model with quantization-aware training."""
    best_accuracy = 0
    patience_counter = 0
    
    logger.info("-" * 60)
    logger.info("Training Configuration:")
    logger.info(f"Batch Size: {Config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {Config.LEARNING_RATE}")
    logger.info(f"Max Epochs: {Config.EPOCHS}")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Model Architecture:\n{str(model)}")
    logger.info("-" * 60)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        
        epoch_start_time = datetime.now()
        logger.info(f"\nEpoch {epoch+1}/{Config.EPOCHS} started at {epoch_start_time.strftime('%H:%M:%S')}")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0:
                avg_loss = total_loss / batch_count
                logger.info(f"Loss: {loss.item():.4f} | "
                            f"Avg Loss: {avg_loss:.4f}")
        
        # Calculate epoch statistics
        epoch_loss = total_loss / len(train_loader)
        accuracy = evaluate_model(model, test_loader)
        epoch_time = datetime.now() - epoch_start_time
        
        # Log epoch summary
        logger.info("\nEpoch Summary:")
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS} completed in {epoch_time}")
        logger.info(f"Average Loss: {epoch_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.2f}%")
        
        if scheduler is not None:
            scheduler.step(accuracy)
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            model_save_path = f'models/best_model_{accuracy:.2f}.pth'
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, model_save_path)
            logger.info(f"New best model saved: {model_save_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        logger.info("-" * 60)

def replace_layers(model: nn.Module, device: str):
    """Replace standard linear layers with quantized versions."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quant_layer = QuantizedLinear(module, 
                                        bit_width=Config.WEIGHT_BITS,
                                        act_bit_width=Config.ACTIVATION_BITS,
                                        device='cuda')
            setattr(model, name, quant_layer)
        else:
            replace_layers(module, device)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    try:
        logger.info("Starting quantization-aware training experiment")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        train_loader, test_loader = get_data_loaders()
        logger.info("Data loaders initialized successfully")
        
        model = torch.load('model-cifar10.pth')
        logger.info("Original model loaded successfully")
        
        # Original model training
        logger.info("\n=== Original Model Training ===")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=2, factor=0.1
        )
        
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
        original_accuracy = evaluate_model(model, test_loader)
        logger.info(f"Original Model Final Accuracy: {original_accuracy:.2f}%")
        
        # Quantized model training
        logger.info("\n=== Quantized Model Training ===")
        qat_model = copy.deepcopy(model)
        replace_layers(qat_model, Config.DEVICE)
        qat_model.to(Config.DEVICE)
        logger.info("Model quantization completed")
        
        optimizer = optim.Adam(qat_model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=2, factor=0.1
        )
        
        train_model(qat_model, train_loader, test_loader, criterion, optimizer, scheduler)
        final_accuracy = evaluate_model(qat_model, test_loader)
        
        # Log final results
        logger.info("\n=== Final Results ===")
        logger.info(f"Original Model Accuracy: {original_accuracy:.2f}%")
        logger.info(f"Quantized Model Accuracy: {final_accuracy:.2f}%")
        logger.info(f"Accuracy Impact: {final_accuracy - original_accuracy:.2f}%")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
