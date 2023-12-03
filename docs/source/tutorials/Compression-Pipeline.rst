===============================
Compression Pipeline
===============================

!pip install sconce -q

.. code:: ipython3

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
    
    assert torch.cuda.is_available(), \
    "The current runtime does not have CUDA support." \
    "Please go to menu bar (Runtime - Change runtime type) and select GPU"

Load the Pre-Trained Model Weights

.. code:: ipython3

    from google.colab import drive
    drive.mount('/content/drive')
    model_path = "drive/MyDrive/Efficientml/Efficientml.ai/vgg.cifar.pretrained.pth"


.. parsed-literal::

    Mounted at /content/drive


.. code:: ipython3

    
    class VGG(nn.Module):
      ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    
      def __init__(self) -> None:
        super().__init__()
    
        layers = []
        counts = defaultdict(int)
    
        def add(name: str, layer: nn.Module) -> None:
          layers.append((f"{name}{counts[name]}", layer))
          counts[name] += 1
    
        in_channels = 3
        for x in self.ARCH:
          if x != 'M':
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
    
    
    #load the pretrained model
    
    model = VGG().cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])




.. parsed-literal::

    <All keys matched successfully>



Setup the Dataset

.. code:: ipython3

    image_size = 32
    transforms = {
        "train": transforms.Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
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
    for split in ['train', 'test']:
      dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
      )
    
    



.. parsed-literal::

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar10/cifar-10-python.tar.gz


.. parsed-literal::

    100%|██████████| 170498071/170498071 [00:10<00:00, 16502419.73it/s]


.. parsed-literal::

    Extracting data/cifar10/cifar-10-python.tar.gz to data/cifar10
    Files already downloaded and verified


sconce Configurations

.. code:: ipython3

    from sconce import sconce
    
    
    sconces = sconce()
    sconces.model= model
    sconces.criterion = nn.CrossEntropyLoss() # Loss
    sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
    sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
    sconces.dataloader = dataloader
    sconces.epochs = 1 #Number of time we iterate over the data
    sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sconces.experiment_name = "vgg-gmp"
    sconces.prune_mode = "GMP" # Supports Automated Pruning Ratio Detection


Train and Validated the Model on the given dataset

.. code:: ipython3

    # Train the model
    sconces.train()
    # Evaludate the model
    sconces.evaluate()


.. parsed-literal::

    

.. parsed-literal::

    Epoch:1 Train Loss: 0.00000 Validation Accuracy: 92.92585


.. parsed-literal::

    



.. parsed-literal::

    92.92585170340682



Magic Happens here: Compress the model(GMP pruning is set as the prune
mode[sconces.prune_mode] above)

Compress the model granuarly
============================

sconces.compress()

**Channel-Wise Pruning**

.. code:: ipython3

    sconces = sconce()
    sconces.model= model
    sconces.criterion = nn.CrossEntropyLoss() # Loss
    sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
    sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
    sconces.dataloader = dataloader
    sconces.epochs = 1 #Number of time we iterate over the data
    sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sconces.experiment_name = "vgg-cwp"
    sconces.prune_mode = "CWP" # Supports Automated Pruning Ratio Detection


.. code:: ipython3

    # Compress the model Channel-Wise
    sconces.compress()


.. parsed-literal::

    
    Original Dense Model Size Model=21.94 MiB


.. parsed-literal::

    

.. parsed-literal::

    Original Model Validation Accuracy: 92.82565130260521 %


.. parsed-literal::

    

.. parsed-literal::

    
     Channel-Wise Pruning
    Sparsity for each Layer: {'backbone.conv0.weight': 0.1, 'backbone.conv1.weight': 0.1, 'backbone.conv2.weight': 0.15000000000000002, 'backbone.conv3.weight': 0.15000000000000002, 'backbone.conv4.weight': 0.25000000000000006, 'backbone.conv5.weight': 0.20000000000000004, 'backbone.conv6.weight': 0.40000000000000013}


.. parsed-literal::

    

.. parsed-literal::

    
    Pruned Model has size=12.81 MiB(non-zeros) = 58.41% of Original model size


.. parsed-literal::

    

.. parsed-literal::

    Epoch:1 Train Loss: 0.00000 Validation Accuracy: 89.83968


.. parsed-literal::

    

.. parsed-literal::

    Epoch:2 Train Loss: 0.00000 Validation Accuracy: 89.97996


.. parsed-literal::

    

.. parsed-literal::

    Epoch:3 Train Loss: 0.00000 Validation Accuracy: 90.21042


.. parsed-literal::

    

.. parsed-literal::

    Epoch:4 Train Loss: 0.00000 Validation Accuracy: 90.39078


.. parsed-literal::

    

.. parsed-literal::

    Epoch:5 Train Loss: 0.00000 Validation Accuracy: 90.59118


.. parsed-literal::

    

.. parsed-literal::

    
     ................. Comparison Table  .................
                    Original        Pruned          Reduction Ratio
    Latency (ms)    3.87            2.12            1.8            
    MACs (M)        606             410             1.5            
    Param (M)       5.75            5.44            1.1            
    Accuracies (%)  92.826          90.591          -2.234         
    Fine-Tuned Sparse model has size=20.76 MiB = 94.62% of Original model size
    Fine-Tuned Pruned Model Validation Accuracy: 90.59118236472946


.. raw:: html

   <h1>

.. raw:: html

   <center>

Benchmarking on RTX4090

.. raw:: html

   </center>

.. raw:: html

   </h1>

-  **Dense Model** has a size of *35.20MiB* and accuracy of *92.89%*.
-  **Post Pruning(GMP) Pruned Model** size *21.94MiB* with accuracy of
   *92.86%*.
-  **Post Pruning(CMP) Pruned Model** size *20.76MiB* with accuracy of
   *90.59%*.

+----------+----------+----------+----------+----------+----------+
| Metric   | Original | CWP      | GMP      | CWP      | GMP      |
|          |          | Pruned   | Pruned   | R        | R        |
|          |          |          |          | eduction | eduction |
|          |          |          |          | Ratio    | Ratio    |
+==========+==========+==========+==========+==========+==========+
| *        | 5.90     | 4.20     | 5.70     | 1.4      | 1        |
| *Latency |          |          |          |          |          |
| (ms)     |          |          |          |          |          |
| [↓]**    |          |          |          |          |          |
+----------+----------+----------+----------+----------+----------+
| **MACs   | 606      | 406      | 606      | 1.5      | 1        |
| (M)      |          |          |          |          |          |
| [↓]**    |          |          |          |          |          |
+----------+----------+----------+----------+----------+----------+
| **Param  | 9.23     | 5.36     | 4.42     | 1.7      | 2.1      |
| (M)[No   |          |          |          |          |          |
| n-Zeros] |          |          |          |          |          |
| [↓]**    |          |          |          |          |          |
+----------+----------+----------+----------+----------+----------+
| **Ac     | 93.136   | 90.391   | 92.946   | -2.745   | -0.19    |
| curacies |          |          |          |          |          |
| (%)      |          |          |          |          |          |
| [↑]**    |          |          |          |          |          |
+----------+----------+----------+----------+----------+----------+

The catch is that GMP stores the zeros in the weight, which contributes
to the higher values of model size.

**Venum Pruning a better version of Wanda Pruning**

.. code:: ipython3

    from sconce import sconce
    
    sconces = sconce()
    sconces.model = model
    sconces.criterion = nn.CrossEntropyLoss()  # Loss
    sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
    sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
    sconces.dataloader = dataloader
    sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sconces.experiment_name = "vgg-venum"
    sconces.prune_mode = "venum"  # Supports Automated Pruning Ratio Detection
    sconces.compress()


.. parsed-literal::

    
    Original Dense Model Size Model=35.20 MiB


.. parsed-literal::

                                                         

.. parsed-literal::

    Original Model Validation Accuracy: 93.13627254509018 %
    
     Venum Pruning


.. parsed-literal::

                                                        

.. parsed-literal::

    Sensitivity Scan Time(secs): 114.05389285087585
    Sparsity for each Layer: {'backbone.conv0.weight': 0.30000000000000004, 'backbone.conv1.weight': 0.45000000000000007, 'backbone.conv2.weight': 0.45000000000000007, 'backbone.conv3.weight': 0.5500000000000002, 'backbone.conv4.weight': 0.6000000000000002, 'backbone.conv5.weight': 0.7000000000000002, 'backbone.conv6.weight': 0.7500000000000002, 'backbone.conv7.weight': 0.8500000000000002, 'classifier.weight': 0.9500000000000003}
    Pruning Time Consumed (secs): 1701416101.321775
    Total Pruning Time Consumed (mins): 2.8907041509946185


.. parsed-literal::

                                                         

.. parsed-literal::

    
    Pruned Model has size=9.94 MiB(non-zeros) = 28.22% of Original model size


.. parsed-literal::

                                                         

.. parsed-literal::

    
     ................. Comparison Table  .................
                    Original        Pruned          Reduction Ratio
    Latency (ms)    5.9             5.8             1.0            
    MACs (M)        606             606             1.0            
    Param (M)       9.23            2.6             3.5            
    Accuracies (%)  93.136          87.735          -5.401         
    Fine-Tuned Sparse model has size=9.94 MiB = 28.22% of Original model size
    Fine-Tuned Pruned Model Validation Accuracy: 87.73547094188376


Spiking Neural Network Compression

.. code:: ipython3

    !pip install snntorch -q

.. code:: ipython3

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
    



.. parsed-literal::

    <ipython-input-3-b898cb6c07c2>:4: DeprecationWarning: The module snntorch.backprop will be deprecated in  a future release. Writing out your own training loop will lead to substantially faster performance.
      from snntorch import backprop


.. code:: ipython3

    
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


.. code:: ipython3

    from sconce import sconce
    sconces = sconce()
    # Set you Dataloader
    dataloader = {}
    dataloader["train"] = train_loader
    dataloader["test"] = test_loader
    sconces.dataloader = dataloader

.. code:: ipython3

    #Enable snn in sconce
    sconces.snn = True
    
    # Load your snn Model
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
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
    ).to('cuda')
    
    
    #Load the pretrained weights
    snn_pretrained_model_path = "drive/MyDrive/Efficientml/Efficientml.ai/snn_model.pth"
    snn_model.load_state_dict(torch.load(snn_pretrained_model_path))  # Model Definition
    sconces.model = snn_model

.. code:: ipython3

    
    sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
    sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
    
    sconces.criterion = SF.ce_rate_loss()
    
    sconces.epochs = 10  # Number of time we iterate over the data
    sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sconces.experiment_name = "snn-gmp"  # Define your experiment name here
    sconces.prune_mode = "GMP"
    sconces.num_finetune_epochs = 1


.. code:: ipython3

    sconces.compress()


.. parsed-literal::

    
    Original Dense Model Size Model=0.11 MiB


.. parsed-literal::

    

.. parsed-literal::

    Original Model Validation Accuracy: 97.11538461538461 %
    Granular-Magnitude Pruning


.. parsed-literal::

    

.. parsed-literal::

    Sparsity for each Layer: {'0.weight': 0.6500000000000001, '3.weight': 0.5000000000000001, '7.weight': 0.7000000000000002}


.. parsed-literal::

    

.. parsed-literal::

    
    Pruned Model has size=0.05 MiB(non-zeros) = 43.13% of Original model size


.. parsed-literal::

    

.. parsed-literal::

    Epoch:1 Train Loss: 0.00000 Validation Accuracy: 95.97356


.. parsed-literal::

    

.. parsed-literal::

    
     ................. Comparison Table  .................
                    Original        Pruned          Reduction Ratio
    Latency (ms)    2.09            1.43            1.5            
    MACs (M)        160             160             1.0            
    Param (M)       0.01            0.01            1.0            
    Accuracies (%)  97.115          95.974          -1.142         
    Fine-Tuned Sparse model has size=0.05 MiB = 43.13% of Original model size
    Fine-Tuned Pruned Model Validation Accuracy: 95.9735576923077


.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/torchprofile/profile.py:22: UserWarning: No handlers found: "prim::pythonop". Skipped.
      warnings.warn('No handlers found: "{}". Skipped.'.format(
    /usr/local/lib/python3.10/dist-packages/torchprofile/profile.py:22: UserWarning: No handlers found: "prim::pythonop". Skipped.
      warnings.warn('No handlers found: "{}". Skipped.'.format(

