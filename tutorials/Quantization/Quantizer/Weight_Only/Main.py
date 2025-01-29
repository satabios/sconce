import torch
from models import VGG, SimpleCNN
from Chunker import Chunker
from tqdm import tqdm
from dataset import Dataset
import copy

@torch.inference_mode()
def evaluate_model(model, test_loader, device ='cuda'):

    model = copy.deepcopy(model).to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
dataloader = Dataset('cifar10')

# model = SimpleCNN()
# model.load_state_dict(torch.load('data/weights/best_model.pth',map_location=torch.device('cpu')))
model = VGG()#.cuda()
checkpoint = torch.load("../../data/weights/vgg.cifar.pretrained.pth", weights_only=True)
model.load_state_dict(checkpoint)
print(f"Original Model Accuracy : {evaluate_model(model, dataloader,device='cuda')}")

# #
# fuser = Fuse(model.eval(), dataloader)
# fused_model = fuser.fused_model.train()
# print(f"Fused Model Accuracy : {evaluate_model(fused_model, dataloader)}")
quantized_model = Chunker(model, dataloader).model
print(quantized_model)
print(f"Quantized Model Accuracy : {evaluate_model(quantized_model, dataloader,device='cpu')}")

