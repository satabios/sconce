.. code:: ipython3

    import torch
    import torchvision.datasets as datasets 
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from torch import optim
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

    image_size = 32
    transforms = {
        "train":  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        "test":  transforms.Compose([transforms.ToTensor()])
    
    }
    dataset = {}
    for split in ["train", "test"]:
      dataset[split] = datasets.MNIST(
        root="data/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms.get(split),
      )
    dataloader = {}
    for split in ['train', 'test']:
      dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
      )

Define the model
================

.. code:: ipython3

    class Net(nn.Module):
        def __init__(self, hidden_size_1=100, hidden_size_2=100):
            super(Net,self).__init__()
            self.linear1 = nn.Linear(28*28, hidden_size_1) 
            self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
            self.linear3 = nn.Linear(hidden_size_2, 10)
            self.relu = nn.ReLU()
    
        def forward(self, img):
            x = img.view(-1, 28*28)
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = self.linear3(x)
            return x

.. code:: ipython3

    net = Net()

Train the model
===============

.. code:: ipython3

    from sconce import sconce
    
    sconces = sconce()
    sconces.model= Net() # Model Definition
    sconces.criterion = nn.CrossEntropyLoss() # Loss
    sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
    sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
    sconces.dataloader = dataloader
    sconces.epochs = 10 #Number of time we iterate over the data
    sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sconces.experiment_name = "ptq" # Define your experiment name here
        
       
    sconces.train()
    sconces.epochs = 10 #Number of time we iterate over the data
    sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sconces.experiment_name = "ptq" # Define your experiment name here
        
       
    sconces.train()


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:1 Train Loss: 0.00000 Validation Accuracy: 92.24449


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:2 Train Loss: 0.00000 Validation Accuracy: 92.28457


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:3 Train Loss: 0.00000 Validation Accuracy: 92.19439


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:4 Train Loss: 0.00000 Validation Accuracy: 92.54509


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:5 Train Loss: 0.00000 Validation Accuracy: 92.96593


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:6 Train Loss: 0.00000 Validation Accuracy: 92.99599


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:7 Train Loss: 0.00000 Validation Accuracy: 93.05611


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:8 Train Loss: 0.00000 Validation Accuracy: 93.50701


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:9 Train Loss: 0.00000 Validation Accuracy: 93.60721


.. parsed-literal::

                                                            

.. parsed-literal::

    Epoch:10 Train Loss: 0.00000 Validation Accuracy: 93.82766


.. parsed-literal::

    

.. code:: ipython3

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp_delme.p")
        print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
        os.remove('temp_delme.p')
    
    MODEL_FILENAME = 'ptq.pth'
    


Define the testing loop
=======================

.. code:: ipython3

    sconces.evaluate()


.. parsed-literal::

                                                         



.. parsed-literal::

    93.82765531062124



Print weights and size of the model before quantization
=======================================================

.. code:: ipython3

    # Print the weights matrix of the model before quantization
    print('Weights before quantization')
    print(sconces.model.linear1.weight)
    print(sconces.model.linear1.weight.dtype)


.. parsed-literal::

    Weights before quantization
    Parameter containing:
    tensor([[ 0.0155,  0.0052,  0.0159,  ...,  0.0197,  0.0095, -0.0005],
            [-0.0068,  0.0042, -0.0486,  ...,  0.0128, -0.0346, -0.0415],
            [-0.0371,  0.0240, -0.0176,  ..., -0.0015, -0.0090, -0.0394],
            ...,
            [-0.0173, -0.0132,  0.0111,  ..., -0.0179, -0.0355,  0.0213],
            [ 0.0144, -0.0423, -0.0032,  ..., -0.0063, -0.0037, -0.0377],
            [ 0.0240, -0.0030,  0.0295,  ...,  0.0229,  0.0326, -0.0246]],
           device='cuda:0', requires_grad=True)
    torch.float32


.. code:: ipython3

    print('Size of the model before quantization')
    print_size_of_model(sconces.model)


.. parsed-literal::

    Size of the model before quantization
    Size (KB): 361.062


.. code:: ipython3

    print(f'Accuracy of the model before quantization: ')
    sconces.evaluate()


.. parsed-literal::

    Accuracy of the model before quantization: 


.. parsed-literal::

                                                         



.. parsed-literal::

    93.82765531062124



Insert min-max observers in the model
=====================================

.. code:: ipython3

    class QuantizedNet(nn.Module):
        def __init__(self, hidden_size_1=100, hidden_size_2=100):
            super(QuantizedNet,self).__init__()
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

.. code:: ipython3

    device = 'cpu'
    net_quantized = QuantizedNet().to(device)
    # Copy weights from unquantized model
    net_quantized.load_state_dict(torch.load('ptq.pth'))
    net_quantized.eval()
    sconces.model = net_quantized
    
    net_quantized.qconfig = torch.ao.quantization.default_qconfig
    sconces.model = torch.ao.quantization.prepare(sconces.model) # Insert observers
    sconces.model




.. parsed-literal::

    QuantizedNet(
      (quant): QuantStub(
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (linear1): Linear(
        in_features=784, out_features=100, bias=True
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (linear2): Linear(
        in_features=100, out_features=100, bias=True
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (linear3): Linear(
        in_features=100, out_features=10, bias=True
        (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
      )
      (relu): ReLU()
      (dequant): DeQuantStub()
    )



Calibrate the model using the test set
======================================

.. code:: ipython3

    sconces.evaluate()


.. parsed-literal::

                                                         



.. parsed-literal::

    93.82765531062124



.. code:: ipython3

    print(f'Check statistics of the various layers')
    sconces.model


.. parsed-literal::

    Check statistics of the various layers




.. parsed-literal::

    QuantizedNet(
      (quant): QuantStub(
        (activation_post_process): MinMaxObserver(min_val=0.0, max_val=1.0)
      )
      (linear1): Linear(
        in_features=784, out_features=100, bias=True
        (activation_post_process): MinMaxObserver(min_val=-6.078974723815918, max_val=5.0761823654174805)
      )
      (linear2): Linear(
        in_features=100, out_features=100, bias=True
        (activation_post_process): MinMaxObserver(min_val=-2.7763729095458984, max_val=4.160463809967041)
      )
      (linear3): Linear(
        in_features=100, out_features=10, bias=True
        (activation_post_process): MinMaxObserver(min_val=-3.7380480766296387, max_val=3.566555976867676)
      )
      (relu): ReLU()
      (dequant): DeQuantStub()
    )



Quantize the model using the statistics collected
=================================================

.. code:: ipython3

    sconces.model = torch.ao.quantization.convert(sconces.model)

.. code:: ipython3

    print(f'Check statistics of the various layers')
    sconces.model


.. parsed-literal::

    Check statistics of the various layers




.. parsed-literal::

    QuantizedNet(
      (quant): Quantize(scale=tensor([0.0079]), zero_point=tensor([0]), dtype=torch.quint8)
      (linear1): QuantizedLinear(in_features=784, out_features=100, scale=0.0878358855843544, zero_point=69, qscheme=torch.per_tensor_affine)
      (linear2): QuantizedLinear(in_features=100, out_features=100, scale=0.054620761424303055, zero_point=51, qscheme=torch.per_tensor_affine)
      (linear3): QuantizedLinear(in_features=100, out_features=10, scale=0.057516567409038544, zero_point=65, qscheme=torch.per_tensor_affine)
      (relu): ReLU()
      (dequant): DeQuantize()
    )



Print weights of the model after quantization
=============================================

.. code:: ipython3

    # Print the weights matrix of the model after quantization
    print('Weights after quantization')
    print(torch.int_repr(sconces.model.linear1.weight()))


.. parsed-literal::

    Weights after quantization
    tensor([[ 14,   5,  15,  ...,  18,   9,   0],
            [ -6,   4, -45,  ...,  12, -32, -38],
            [-34,  22, -16,  ...,  -1,  -8, -36],
            ...,
            [-16, -12,  10,  ..., -17, -33,  20],
            [ 13, -39,  -3,  ...,  -6,  -3, -35],
            [ 22,  -3,  27,  ...,  21,  30, -23]], dtype=torch.int8)


Compare the dequantized weights and the original weights
========================================================

.. code:: ipython3

    print('Original weights: ')
    print(net.linear1.weight)
    print('')
    print(f'Dequantized weights: ')
    print(torch.dequantize(sconces.model.linear1.weight()))
    print('')


.. parsed-literal::

    Original weights: 
    Parameter containing:
    tensor([[-0.0003,  0.0192, -0.0294,  ...,  0.0219,  0.0037,  0.0021],
            [-0.0198, -0.0150, -0.0104,  ..., -0.0203, -0.0060, -0.0299],
            [-0.0201,  0.0149, -0.0333,  ..., -0.0203,  0.0012,  0.0080],
            ...,
            [ 0.0221,  0.0258, -0.0088,  ..., -0.0141,  0.0051, -0.0318],
            [-0.0217, -0.0136,  0.0185,  ..., -0.0012, -0.0012, -0.0017],
            [ 0.0142,  0.0089, -0.0053,  ...,  0.0311, -0.0181,  0.0020]],
           requires_grad=True)
    
    Dequantized weights: 
    tensor([[ 0.0151,  0.0054,  0.0162,  ...,  0.0195,  0.0097,  0.0000],
            [-0.0065,  0.0043, -0.0487,  ...,  0.0130, -0.0346, -0.0411],
            [-0.0368,  0.0238, -0.0173,  ..., -0.0011, -0.0086, -0.0389],
            ...,
            [-0.0173, -0.0130,  0.0108,  ..., -0.0184, -0.0357,  0.0216],
            [ 0.0141, -0.0422, -0.0032,  ..., -0.0065, -0.0032, -0.0378],
            [ 0.0238, -0.0032,  0.0292,  ...,  0.0227,  0.0324, -0.0249]])
    


Print size and accuracy of the quantized model
==============================================

.. code:: ipython3

    print('Size of the model after quantization')
    print_size_of_model(sconces.model)


.. parsed-literal::

    Size of the model after quantization
    Size (KB): 95.394


.. code:: ipython3

    print('Testing the model after quantization')
    sconces.model.to('cpu')
    sconces.evaluate()


.. parsed-literal::

    Testing the model after quantization


.. parsed-literal::

                                                         



.. parsed-literal::

    93.937875751503


