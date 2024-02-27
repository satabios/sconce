Import the necessary libraries
==============================

.. code:: ipython3

    import torch
    import torchvision.datasets as datasets 
    import torchvision.transforms as transforms
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from pathlib import Path
    import os


.. parsed-literal::

    /home/sathya/anaconda3/envs/zen/lib/python3.11/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      _torch_pytree._register_pytree_node(


Load the MNIST dataset
======================

.. code:: ipython3

    # Make torch deterministic
    _ = torch.manual_seed(0)

.. code:: ipython3

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Load the MNIST dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Create a dataloader for the training
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
    
    # Load the MNIST test set
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)
    
    # Define the device
    device = "cpu"

Define the model
================

.. code:: ipython3

    class Net(nn.Module):
        def __init__(self, hidden_size_1=100, hidden_size_2=100):
            super(Net,self).__init__()
            self.quant = torch.quantization.QuantStub()
            self.linear1 = nn.Linear(28*28, hidden_size_1) 
            self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
            self.linear3 = nn.Linear(hidden_size_2, 10)
            self.relu = nn.ReLU()
            self.dequant = torch.quantization.DeQuantStub()
    
        def forward(self, img):
            x = img.view(-1, 28*28)
            x = self.quant(x)
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = self.linear3(x)
            x = self.dequant(x)
            return x
    
    net = Net().to(device)

Insert min-max observers in the model
=====================================

.. code:: ipython3

    net.qconfig = torch.ao.quantization.default_qconfig
    net.train()
    net_quantized = torch.ao.quantization.prepare_qat(net) # Insert observers
    net_quantized




.. parsed-literal::

    Net(
      (quant): QuantStub(
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (linear1): Linear(
        in_features=784, out_features=100, bias=True
        (weight_fake_quant): MinMaxObserver(min_val=inf, max_val=-inf)
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (linear2): Linear(
        in_features=100, out_features=100, bias=True
        (weight_fake_quant): MinMaxObserver(min_val=inf, max_val=-inf)
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (linear3): Linear(
        in_features=100, out_features=10, bias=True
        (weight_fake_quant): MinMaxObserver(min_val=inf, max_val=-inf)
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (relu): ReLU()
      (dequant): DeQuantStub()
    )



Train the model
===============

.. code:: ipython3

    def train(train_loader, net, epochs=5, total_iterations_limit=None):
        cross_el = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
        total_iterations = 0
    
        for epoch in range(epochs):
            net.train()
    
            loss_sum = 0
            num_iterations = 0
    
            data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            if total_iterations_limit is not None:
                data_iterator.total = total_iterations_limit
            for data in data_iterator:
                num_iterations += 1
                total_iterations += 1
                x, y = data
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                output = net(x.view(-1, 28*28))
                loss = cross_el(output, y)
                loss_sum += loss.item()
                avg_loss = loss_sum / num_iterations
                data_iterator.set_postfix(loss=avg_loss)
                loss.backward()
                optimizer.step()
    
                if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                    return
                
    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp_delme.p")
        print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
        os.remove('temp_delme.p')
    
    train(train_loader, net_quantized, epochs=1)


.. parsed-literal::

    Epoch 1: 100%|██████████| 6000/6000 [00:19<00:00, 308.90it/s, loss=0.218]


Define the testing loop
=======================

.. code:: ipython3

    def test(model: nn.Module, total_iterations: int = None):
        correct = 0
        total = 0
    
        iterations = 0
    
        model.eval()
    
        with torch.no_grad():
            for data in tqdm(test_loader, desc='Testing'):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                output = model(x.view(-1, 784))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct +=1
                    total +=1
                iterations += 1
                if total_iterations is not None and iterations >= total_iterations:
                    break
        print(f'Accuracy: {round(correct/total, 3)*100}')

Check the collected statistics during training
==============================================

.. code:: ipython3

    print(f'Check statistics of the various layers')
    net_quantized


.. parsed-literal::

    Check statistics of the various layers




.. parsed-literal::

    VerySimpleNet(
      (quant): Quantize(scale=tensor([0.0256]), zero_point=tensor([17]), dtype=torch.quint8)
      (linear1): QuantizedLinear(in_features=784, out_features=100, scale=0.5840995907783508, zero_point=69, qscheme=torch.per_tensor_affine)
      (linear2): QuantizedLinear(in_features=100, out_features=100, scale=0.4839491844177246, zero_point=80, qscheme=torch.per_tensor_affine)
      (linear3): QuantizedLinear(in_features=100, out_features=10, scale=0.4375518560409546, zero_point=75, qscheme=torch.per_tensor_affine)
      (relu): ReLU()
      (dequant): DeQuantize()
    )



Quantize the model using the statistics collected
=================================================

.. code:: ipython3

    net_quantized.eval()
    net_quantized = torch.ao.quantization.convert(net_quantized)

.. code:: ipython3

    print(f'Check statistics of the various layers')
    net_quantized


.. parsed-literal::

    Check statistics of the various layers




.. parsed-literal::

    VerySimpleNet(
      (quant): Quantize(scale=tensor([0.0256]), zero_point=tensor([17]), dtype=torch.quint8)
      (linear1): QuantizedLinear(in_features=784, out_features=100, scale=0.5840995907783508, zero_point=69, qscheme=torch.per_tensor_affine)
      (linear2): QuantizedLinear(in_features=100, out_features=100, scale=0.4839491844177246, zero_point=80, qscheme=torch.per_tensor_affine)
      (linear3): QuantizedLinear(in_features=100, out_features=10, scale=0.4375518560409546, zero_point=75, qscheme=torch.per_tensor_affine)
      (relu): ReLU()
      (dequant): DeQuantize()
    )



Print weights and size of the model after quantization
======================================================

.. code:: ipython3

    # Print the weights matrix of the model before quantization
    print('Weights before quantization')
    print(torch.int_repr(net_quantized.linear1.weight()))


.. parsed-literal::

    Weights before quantization
    tensor([[ 3,  8, -4,  ...,  9,  4,  4],
            [-6, -5, -4,  ..., -6, -3, -9],
            [ 0,  8, -3,  ...,  0,  5,  6],
            ...,
            [10, 11,  3,  ...,  2,  6, -3],
            [-2,  0,  8,  ...,  3,  3,  3],
            [ 5,  4,  0,  ...,  9, -3,  2]], dtype=torch.int8)


.. code:: ipython3

    print('Testing the model after quantization')
    test(net_quantized)


.. parsed-literal::

    Testing the model after quantization


.. parsed-literal::

    Testing: 100%|██████████| 1000/1000 [00:01<00:00, 684.23it/s]

.. parsed-literal::

    Accuracy: 95.3


.. parsed-literal::

    

