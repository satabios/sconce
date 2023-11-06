===============================
 sconce Pipeline Tutorial
===============================
.. code:: ipython3

    %pip install sconce -q
    %pip show sconce
    
    
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


.. parsed-literal::

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m153.1/153.1 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Installing backend dependencies ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.6/11.6 MB[0m [31m107.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.2/18.2 MB[0m [31m64.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.6/3.6 MB[0m [31m76.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m109.0/109.0 kB[0m [31m14.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.9/7.9 MB[0m [31m124.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.0/302.0 kB[0m [31m39.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.8/3.8 MB[0m [31m117.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m89.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.1/53.1 kB[0m [31m7.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m86.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m295.0/295.0 kB[0m [31m35.5 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for lit (pyproject.toml) ... [?25l[?25hdone
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    lida 0.0.10 requires fastapi, which is not installed.
    lida 0.0.10 requires kaleido, which is not installed.
    lida 0.0.10 requires python-multipart, which is not installed.
    lida 0.0.10 requires uvicorn, which is not installed.
    cupy-cuda11x 11.0.0 requires numpy<1.26,>=1.20, but you have numpy 1.26.1 which is incompatible.
    imageio 2.31.6 requires pillow<10.1.0,>=8.3.2, but you have pillow 10.1.0 which is incompatible.
    numba 0.56.4 requires numpy<1.24,>=1.18, but you have numpy 1.26.1 which is incompatible.
    tensorflow-probability 0.22.0 requires typing-extensions<4.6.0, but you have typing-extensions 4.8.0 which is incompatible.[0m[31m
    [0mName: sconce
    Version: 0.57
    Summary: sconce: torch pipeliner  
    Home-page: https://github.com/satabios/sconce
    Author: Sathyaprakash Narayanan
    Author-email: Sathyaprakash Narayanan <snaray17@ucsc.edu>
    License: 
    Location: /usr/local/lib/python3.10/dist-packages
    Requires: certifi, charset-normalizer, cmake, contourpy, cycler, fast-pytorch-kmeans, filelock, fonttools, idna, ipdb, Jinja2, kiwisolver, lit, MarkupSafe, matplotlib, mpmath, networkx, numpy, packaging, Pillow, pyparsing, python-dateutil, requests, six, snntorch, sympy, torch, torchprofile, torchvision, tqdm, transformers, typing-extensions, urllib3
    Required-by: 


Load the Pre-Trained Model Weights

.. code:: ipython3

    from google.colab import drive
    drive.mount('/content/drive')
    model_path = "drive/MyDrive/Efficientml/Efficientml.ai/pre-trained_vgg.cifar.pretrained.pth"


.. parsed-literal::

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


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

    100%|██████████| 170498071/170498071 [00:12<00:00, 13129669.34it/s]


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

    train:   0%|          | 0/98 [00:00<?, ?it/s]



.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]


.. parsed-literal::

    Epoch:1 Train Loss: 0.00000 Validation Accuracy: 92.89579



.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]




.. parsed-literal::

    92.89579158316633



Magic Happens here: Compress the model(GMP pruning is set as the prune
mode[sconces.prune_mode] above)

.. code:: ipython3

    # Compress the model
    sconces.compress()


.. parsed-literal::

    
    Dense_model_size model after sensitivity size=35.20 MiB



.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]


.. parsed-literal::

    Original Model Validation Accuracy: 92.89579158316633 %
    Granular-Magnitude Pruning


.. parsed-literal::

    Sparsity for each Layer: {'backbone.conv0.weight': 0.20000000000000004, 'backbone.conv1.weight': 0.30000000000000004, 'backbone.conv2.weight': 0.1, 'backbone.conv3.weight': 0.3500000000000001, 'backbone.conv4.weight': 0.3500000000000001, 'backbone.conv5.weight': 0.3500000000000001, 'backbone.conv6.weight': 0.3500000000000001, 'backbone.conv7.weight': 0.3500000000000001, 'classifier.weight': 0.6500000000000001}
    Pruned model has size=23.18 MiB = 65.85% of Original model size



.. parsed-literal::

    train:   0%|          | 0/98 [00:00<?, ?it/s]



.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]


.. parsed-literal::

    Epoch:1 Train Loss: 0.00000 Validation Accuracy: 93.27655



.. parsed-literal::

    train:   0%|          | 0/98 [00:00<?, ?it/s]

.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]

.. parsed-literal::

    Epoch:2 Train Loss: 0.00000 Validation Accuracy: 93.13627

.. parsed-literal::

    train:   0%|          | 0/98 [00:00<?, ?it/s]

.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]


.. parsed-literal::

    Epoch:3 Train Loss: 0.00000 Validation Accuracy: 93.22645

.. parsed-literal::

    train:   0%|          | 0/98 [00:00<?, ?it/s]

.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]

.. parsed-literal::

    Epoch:4 Train Loss: 0.00000 Validation Accuracy: 93.16633

.. parsed-literal::

    train:   0%|          | 0/98 [00:00<?, ?it/s]

.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]

.. parsed-literal::

    Epoch:5 Train Loss: 0.00000 Validation Accuracy: 93.19639

.. parsed-literal::

    test:   0%|          | 0/20 [00:00<?, ?it/s]


.. parsed-literal::

                    Original        Pruned          Reduction Ratio
    Latency (ms)    19900.0         19900.0         1.0            
    MACs (M)        606             606             1.0            
    Param (M)       9.23            9.23            1.0            
    Fine-Tuned Sparse model has size=23.18 MiB = 65.85% of Original model size
    Fine-Tuned Pruned Model Validation Accuracy: 93.19639278557115


Notice that intially,

-  **Dense Model** has a size of *35.20MiB* and accuracy of *92.89%*.
-  **Post Pruning(GMP) Pruned Model** size *23.18MiB* with accuracy of
   *65.85%*.
-  Upon **fine-tuning the Prune Model**, we have the final pruned model
   size of *23.18MiB* with an accuracy of *93.19%*.

+---------------------+----------+-----------+-----------------+
| Metric              | Original | Pruned    | Reduction Ratio |
+=====================+==========+===========+=================+
| Latency (ms)        | 19900.0  | 19900.0   | 1.0             |
+---------------------+----------+-----------+-----------------+
| MACs (M)            | 606      | 606       | 1.0             |
+---------------------+----------+-----------+-----------------+
| Param (M)           | 9.23     | 9.23      | 1.0             |
+---------------------+----------+-----------+-----------------+
| Fine-Tuned Sparse   | -        | 23.18 MiB | 65.85%          |
| Model Size          |          |           |                 |
+---------------------+----------+-----------+-----------------+
| Fine-Tuned Pruned   | -        | 93.196%   | -               |
| Model Validation    |          |           |                 |
| Accuracy            |          |           |                 |
+---------------------+----------+-----------+-----------------+
