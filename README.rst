================
Introduction
================

.. image:: https://readthedocs.org/projects/sconce/badge/?version=latest
        :target: https://sconce.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://github.com/satabios/sconce/blob/main/docs/source/images/sconce-punch-bk_removed.png?raw=true
        :align: center
        :width: 400
        :height: 400

Advancement of deep learning has been largely driven by the availability of large datasets and the computational power to train large models.
The amount of complexity increases with each day passing and it is becoming increasingly difficult to train these models. Neverthless, infer 
the models efficiently on hardware.

However, the brain is able to learn from a few examples and is extremely energy efficient(Psst.. that too sparsely). Humans tend solve problems from their lens of perspective,
and thus we comprehend the universe through mathematical models. One such ideation is the concept of gradient descent or other optimization techniques
that we use to train our models. However, the brain does not use gradient descent to learn. It is still a mystery how the brain learns and how it is able to
solve complex problems with such ease.


.. image:: https://github.com/satabios/sconce/blob/main/docs/source/images/sconce-pipeline.png?raw=true
        :align: center
       


To bridge this gap, this package aids to perform a series of aids:

* Make **Training**, **Testing**, **Inference**, **Model Profiling**, etc.. pipelined. Thus easing your way through research and development.
* **Compress** the model through **Pruning**, **Optimal Brain Compression**, etc... This allows lesser usage of CPM(Computation, Power, Memory) and thus making it more efficient.
* **Quantize** the model to make it more efficient for hardware Deployment/Inferences.
* Leverage **Sparsity** in the model to make it more efficient for hardware Deployment/Inferences. 
* **Deployments** of the model on hardware.
* Support `Spiking Neural Networks(snnTorch) <https://github.com/jeshraghian/snntorch>`_ in this compression pipeline [Future integerations are expected].
* **Auto-Sensitivity Scan**: Each model would require a set of ingredients of its own to make it efficient. sconce enables an auto-search algorithm that picks the best possible solution from a corpus amount of possible techniques in the fastest manner possible with the least amount of human intervention.


If you like this project, please consider starring ⭐ this repo as it is the easiest and best way to support it.

Let us know if you are using sconce in any interesting work, research or blogs, as we would love to hear more about it! 
If you have issues, comments, or are looking for advice on training spiking neural networks, you can open an issue, a discussion, 
or chat in our `discord <https://discord.gg/GKwXMrZr>`_ channel.

sconce Structure
^^^^^^^^^^^^^^^^^^^^^^^^
sconce contains the following components: 

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Component
     - Description
   * - `sconce.train <https://sconce.readthedocs.io/en/latest/usage.html#module-sconce>`_
     - a spiking neuron library like torch.nn, deeply integrated with autograd
   * - `sconce.measure_latency <https://sconce.readthedocs.io/en/latest/usage.html#module-sconce>`_
     - Compares the performance of two PyTorch models: an original dense model and a pruned and fine-tuned model. Prints a table of metrics including latency, MACs, and model size for both models and their reduction ratios.
   * - `sconce.prune_mode <https://sconce.readthedocs.io/en/latest/usage.html#module-sconce>`_
     - Currently supporting Gradual Magnitude Pruning(GMP), L1/L2 based Channel Wise Pruning(CWP), OBC, sparsegpt, etc...
   * - `sconce.quantize <https://sconce.readthedocs.io/en/latest/usage.html#module-sconce>`_
     - Quantize the computations of the model to make it more efficient for hardware Deployment/Inferences.
   * - `sconce.compress <https://sconce.readthedocs.io/en/latest/usage.html#module-sconcel>`_
     - Automated compression pipeline encompassing of Pruning, Quantization, and Sparsification.
  
**sconce** is designed to be intuitively used with PyTorch, compression for Linear, Convolutional and Attention blocks are supported.

At present, we are working on adding support for more compression techniques and more models.
The package envisions to be a one stop solution for all your compression needs and deployed on resource constrained devices.
Provided that the network models and tensors are loaded onto CUDA, sconce takes advantage of GPU acceleration in the same way as PyTorch. 

sconce is a work in progress, and we welcome contributions from the community.

Requirements 
^^^^^^^^^^^^^^^^^^^^^^^^
The following packages need to be installed to use sconce:

* torch >= 1.1.0
* numpy >= 1.17
* torchprofile
* matplotlib
* snntorch

They are automatically installed if sconce is installed using the pip command. Ensure the correct version of torch is installed for your system to enable CUDA compatibility. 

Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Run the following to install:

.. code-block:: bash

  $ python
  $ pip install sconce

To install sconce from source instead::

  $ git clone https://github.com/satabios/sconce
  $ cd sconce
  $ python setup.py install
    

API & Examples 
^^^^^^^^^^^^^^^^^^^^^^^^
A complete API is available `here <https://sconce.readthedocs.io/>`_. Examples, tutorials and Colab notebooks are provided.


Quickstart 
^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/satabios/sconce/blob/main/tutorials/Compression%20Pipeline.ipynb#


Here are a few ways you can get started with sconce:


* `Quickstart Notebook (Opens in Colab)`_

* `The API Reference`_ 

* `Tutorials`_

.. _Quickstart Notebook (Opens in Colab): https://colab.research.google.com/github/satabios/sconce/blob/main/tutorials/Compression%20Pipeline.ipynb
.. _The API Reference: https://sconce.readthedocs.io/
.. _Tutorials: https://sconce.readthedocs.io/en/latest/tutorials/index.html

Quickstart:
================


Define Network:
----------------------------


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
----------------------------

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


Contributing
^^^^^^^^^^^^^^^^^^^^^^^^

If you're ready to contribute to sconce, ping on `discord <https://discord.gg/GKwXMrZr>`_ channel.

Acknowledgments
^^^^^^^^^^^^^^^^^^^^^^^^

sconce is solely being maintained by `Sathyaprakash Narayanan <https://satabios.github.io/portfolio/>`_.

Special Thanks:

*  `Prof. and Mentor Jason K. Eshraghian <https://www.jasoneshraghian.com/>`_ and his pet `snnTorch <https://github.com/jeshraghian/snntorch/>`_ (extensively inspired from snnTorch to build and document sconce)
*  `Prof. Song Han <https://hanlab.mit.edu/>`_ for his coursework MIT6.5940 and many other projects like `torchsparse <https://github.com/mit-han-lab/torchsparse/>`_. 
*  `Neural Magic(Elias Frantar, Denis Kuznedelev, etc...) <https://github.com/neuralmagic/>`_ for `OBC <https://github.com/IST-DASLab/OBC/>`_ and `sparseGPT <https://github.com/IST-DASLab/sparsegpt/>`_.


License & Copyright
^^^^^^^^^^^^^^^^^^^^^^^^

sconce source code is published under the terms of the MIT License. 
sconce's documentation is licensed under a Creative Commons Attribution-Share Alike 3.0 Unported License (`CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>`_).
