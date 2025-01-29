import torch
from models import VGG, SimpleCNN
from Chunker import Chunker
from tqdm import tqdm
from dataset import Dataset
import copy
from Fusion import Fuse

@torch.inference_mode()
def evaluate(
  model,
  dataloader,
  extra_preprocess = None,
        device = None
) -> float:
    model = model.to(device)
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
        inputs = inputs.to(device)
        if extra_preprocess is not None:
            for preprocess in extra_preprocess:
                inputs = preprocess(inputs)

    targets = targets.to(device)
    outputs = model(inputs)
    outputs = outputs.argmax(dim=1)
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()



class cfg:
    model_type =   "vgg" # "vgg" # "simplecnn" # "resnet

if(cfg.model_type == "simplecnn"):
    dataloader = Dataset('cifar10-simplecnn')
    model = SimpleCNN()
    model.load_state_dict(torch.load('data/weights/best_model.pth',map_location=torch.device('cpu'),weights_only=False))

elif(cfg.model_type == "vgg"):
    dataloader = Dataset('cifar10-vgg')
    model = VGG()
    checkpoint = torch.load("data/weights/vgg.cifar.pretrained.pth", weights_only=True)
    model.load_state_dict(checkpoint)

elif(cfg.model_type == "resnet"):
    dataloader = Dataset('imagenet')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
   
print(f"Original Model Accuracy : {evaluate(model, dataloader,device='cuda')}")


# # #
# fuser = Fuse(copy.deepcopy(model.eval()), dataloader)
# fused_model = fuser.fused_model.train()
# print(f"Fused Model Accuracy : {evaluate(fused_model, dataloader, device='cuda')}")
quantized_model = Chunker(copy.deepcopy(model), dataloader).model
del model
# del fused_model
print(quantized_model)



# def extra_preprocess(x):
#     # hint: you need to convert the original fp32 input of range (0, 1)
#     #  into int8 format of range (-128, 127)

#     return (x * 255 - 128).clamp(-128, 127).to(torch.int8)

print(f"Quantized Model Accuracy : {evaluate(quantized_model, dataloader, device='cpu')}")




# int8_model_accuracy = evaluate(quantized_model, dataloader,device='cuda')
# print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")
# torch.quantization.fuse_modules