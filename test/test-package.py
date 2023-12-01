import copy
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import torch.optim as optim

assert torch.cuda.is_available(), (
    "The current runtime does not have CUDA support."
    "Please go to menu bar (Runtime - Change runtime type) and select GPU"
)


class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != "M":
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x


# load the pretrained model

model = VGG().cuda()
checkpoint = torch.load("/home/sathya/Downloads/vgg.cifar.pretrained.pth")
model.load_state_dict(checkpoint["state_dict"])


image_size = 32
transforms = {
    "train": transforms.Compose(
        [
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    ),
    "test": ToTensor(),
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="data/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )

dataloader = {}
for split in ["train", "test"]:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
    )


from sconcevenum import sconce

print("\n=======================================================================")
print("=======================================================================\n")

sconces = sconce()
sconces.model = copy.deepcopy(model)
sconces.criterion = nn.CrossEntropyLoss()  # Loss
sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
sconces.dataloader = dataloader
sconces.epochs = 1  # Number of time we iterate over the data
sconces.num_finetune_epochs = 1
sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sconces.experiment_name = "cwp-gmp"
sconces.prune_mode = "GMP"  # Supports Automated Pruning Ratio Detection
# Compress the model
sconces.compress()

print("\n=======================================================================")
print("=======================================================================\n")

sconces = sconce()
sconces.model = copy.deepcopy(model)
sconces.criterion = nn.CrossEntropyLoss()  # Loss
sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
sconces.dataloader = dataloader
sconces.epochs = 1  # Number of time we iterate over the data
sconces.num_finetune_epochs = 1
sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sconces.experiment_name = "vgg-cwp"
sconces.prune_mode = "CWP"  # Supports Automated Pruning Ratio Detection
# Compress the model
sconces.compress()


print("\n=======================================================================")
print("=======================================================================\n")


sconces = sconce()
sconces.model = copy.deepcopy(model)
sconces.criterion = nn.CrossEntropyLoss()  # Loss
sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
sconces.dataloader = dataloader
sconces.epochs = 1  # Number of time we iterate over the data
sconces.num_finetune_epochs = 1
sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sconces.experiment_name = "vgg-venum"
self.fine_tune=True
sconces.prune_mode = "venum"  # Supports Automated Pruning Ratio Detection
sconces.compress()
