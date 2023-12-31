Usage
=====


.. _installation:

Installation
------------

Run the following to install:

.. code-block:: bash

  $ python
  $ pip install sconce

To install sconce from source instead::

  $ git clone https://github.com/satabios/sconce
  $ cd sconce
  $ python setup.py install


To install sconce with conda::

    $ conda install -c conda-forge sconce
    

Quick-Start
------------


Define Network:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python


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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python


   sconces.compress()