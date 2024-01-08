# Import snntorch libraries
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
from torch import optim

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools


# Event Drive Data

# dataloader arguments
batch_size = 128
data_path = "./data/mnist"

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
)

from sconce import sconce

sconces = sconce()

sconces.snn = True

# Set you Dataloader
dataloader = {}
dataloader["train"] = train_loader
dataloader["test"] = test_loader
sconces.dataloader = dataloader

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
# Load your snn Model
snn_model = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Conv2d(12, 64, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
).to(device)
snn_pretrained_model_path = "/home/sathya/Downloads/snn/snn_model.pth"
snn_model.load_state_dict(torch.load(snn_pretrained_model_path))  # Model Definition
sconces.model = snn_model

sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)

sconces.criterion = SF.ce_rate_loss()

sconces.epochs = 10  # Number of time we iterate over the data
sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sconces.experiment_name = "snn-gmp"  # Define your experiment name here
sconces.prune_mode = "GMP"
sconces.num_finetune_epochs = 1
sconces.compress()
