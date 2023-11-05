====================================
sconce (Model Compression Made Easy)
====================================


.. image:: https://readthedocs.org/projects/sconce/badge/?version=latest
        :target: https://sconce.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://github.com/satabios/sconce/blob/master/docs/source/images/sconce-punch-bk_removed.png?raw=true
        :align: center
        :width: 400

.. raw:: html

   </p>

This is a Pytorch Helper package aimed to aid the workflow of deep
learning model development, compression and deployment.

- This packages has boiler plate defintions that can ease the development of torch model development
- This package has a set of compression techniques that can be used to compress the model

NOTE:
      * Pruning Techniques are imported from Tomoco Package(deperecated)
      * Model Quantization and Deployment features are in the development pipeline which will be available for use soon. 
         
Package install:
================
.. code:: python

   pip install sconce



Example:
========

Define Network and Config’s:
----------------------------

.. code:: python

   # Define your network

   class Net(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 8, 3)
           self.bn1 = nn.BatchNorm2d(8)
           self.pool = nn.MaxPool2d(2, 2)
           self.conv2 = nn.Conv2d(8, 16, 3)
           self.bn2 = nn.BatchNorm2d(16)
           self.fc1 = nn.Linear(16*6*6, 32)
           self.fc2 = nn.Linear(32, 10)

       def forward(self, x):
           x = self.pool(self.bn1(F.relu(self.conv1(x))))
           x = self.pool(self.bn2(F.relu(self.conv2(x))))
           x = torch.flatten(x, 1)
           x = F.relu(self.fc1(x))
           x = self.fc2(x)
           return x

Make a Dict for Dataloader
--------------------------

.. code:: python


   image_size = 32
   transforms = {
       "train": Compose([
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

Define your Configurations:
---------------------------

.. code:: python

   # Define all parameters 

   from sconce import sconce

   sconces = sconce()
   sconces.model= Net() # Model Definition
   sconces.criterion = nn.CrossEntropyLoss() # Loss
   sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
   sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
   sconces.dataloader = dataloader
   sconces.epochs = 5 #Number of time we iterate over the data
   sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   sconces.experiment_name = "vgg-gmp" # Define your experiment name here
   sconces.prune_mode = "GMP" # Prune Mode: Currently supporting "GMP"(Supports Automated Pruning Ratio Detection), "CWP". Future supports for "OBC" and "sparseGPT"



One Roof Solution [Train -> Compress -> Deploy]:
------------------------------------------------

.. code:: python


   sconces.compress()

To-Do
~~~~~

-  ☒ Universal Channel-Wise Pruning

-  ☒ Update Tutorials

-  ☒ Fine Grained Purning

-  ☒ Channel Wise Purning

-  [+] OBC Compression (In-Progress)

-  [+] Spasegpt like Pruning (In-Progress)

-  [+] Quantisation (In-Progress)

-  ☐ Universal AutoML package

-  ☐ Introduction of Sparsification in Pipeline
