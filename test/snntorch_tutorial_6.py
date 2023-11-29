# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools


# Leaky neuron model, overriding the backward pass with a custom function
class LeakySigmoidSurrogate(nn.Module):
    def __init__(self, beta, threshold=1.0, k=25):
        # Leaky_Surrogate is defined in the previous tutorial and not used here
        super(Leaky_Surrogate, self).__init__()

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.surrogate_func = self.FastSigmoid.apply

    # the forward function is called each time we call Leaky
    def forward(self, input_, mem):
        spk = self.surrogate_func((mem - self.threshold))  # call the Heaviside function
        reset = (spk - self.threshold).detach()
        mem = self.beta * mem + input_ - reset
        return spk, mem

    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with gradient of fast sigmoid
    @staticmethod
    class FastSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem, k=25):
            ctx.save_for_backward(
                mem
            )  # store the membrane potential for use in the backward pass
            ctx.k = k
            out = (mem > 0).float()  # Heaviside on the forward pass: Eq(1)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors  # retrieve membrane potential
            grad_input = grad_output.clone()
            grad = (
                grad_input / (ctx.k * torch.abs(mem) + 1.0) ** 2
            )  # gradient of fast sigmoid on backward pass: Eq(4)
            return grad, None


spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)


# dataloader arguments
batch_size = 128
data_path = "/tmp/data/mnist"

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


# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3, mem3


#  Initialize Network
original_net = nn.Sequential(
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

#  Initialize Network
pruned_net = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Conv2d(12, 64, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
).cuda()


def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


# already imported snntorch.functional as SF
# loss_fn =


def batch_accuracy(train_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


# test_acc = batch_accuracy(test_loader, net, num_steps)

# print(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")


# optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
# num_epochs = 1
# loss_hist = []
# test_acc_hist = []
counter = 0


# net.load_state_dict(torch.load('snn_model.pth'))
# Outer training loop
# for epoch in range(num_epochs):
#
#     # Training loop
#     for data, targets in iter(train_loader):
#         data = data.to(device)
#         targets = targets.to(device)
#
#         # forward pass
#         net.train()
#         spk_rec, _ = forward_pass(net, num_steps, data)
#
#         # initialize the loss & sum over time
#         loss_val = loss_fn(spk_rec, targets)
#
#         # Gradient calculation + weight update
#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()
#
#         # Store loss history for future plotting
#         loss_hist.append(loss_val.item())
#
#         # Test set
#         if counter % 50 == 0:
#           with torch.no_grad():
#               net.eval()
#
#               # Test set forward pass
#               test_acc = batch_accuracy(test_loader, net, num_steps)
#               print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
#               test_acc_hist.append(test_acc.item())
#
#         counter += 1
#
#
#


# Define all parameters
from torch import optim
from sconce import sconce


dataloader = {}
dataloader["train"] = train_loader
dataloader["test"] = test_loader

sconces = sconce()
original_net.load_state_dict(torch.load("snn_model.pth"))  # Model Definition
sconces.model = original_net
sconces.criterion = SF.ce_rate_loss()
# sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
# sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
sconces.dataloader = dataloader
sconces.snn = True
sconces.epochs = 1  # Number of time we iterate over the data
sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sconces.experiment_name = "snn-gmp"  # Define your experiment name here
sconces.prune_mode = "GMP"
# sconces.compress()

#
# original_net.load_state_dict(torch.load('snn-gmp.pth'))
# original_net.eval()
pruned_net.load_state_dict(torch.load("snn-gmp_pruned.pth"))
sconces.compare_models(sconces.model, pruned_net)
# sconces.compress()
