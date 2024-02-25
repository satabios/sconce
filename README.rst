================
Introduction
================

.. image:: https://readthedocs.org/projects/sconce/badge/?version=latest
        :target: https://sconce.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
.. image:: https://github.com/satabios/sconce/actions/workflows/python-publish.yml/badge.svg
        :target: https://github.com/satabios/sconce/actions/workflows/python-publish.yml
        :alt: Python Package


The advancement of deep learning has been primarily driven by the availability of large datasets and the computational power to train large models. The complexity increases with each day, and it is becoming increasingly difficult to train/deploy these models.

However, the brain can learn from a few examples and is incredibly energy efficient (Psst.. that too sparsely).

Humans tend to solve problems from their lens of perspective, and thus, we comprehend the universe through mathematical models. One such postulation is the concept of gradient descent or other optimization techniques we use to train our models. However, the brain does not use gradient descent to learn. How the brain learns and can solve complex problems is still a mystery.

Until we hit the holy grail, we must make wise methods to achieve efficiency. **The logical solution is to minimize the usage of CPM (Computation, Power, Memory) and thus achieve high throughputs and latency gains.**

Hence, this package aims to bridge this gap by compressing a model end-to-end and making it hardware-friendly with Minimal Human Intervention.

|

.. image:: https://github.com/satabios/sconce/blob/main/docs/source/images/sconce-features.jpg?raw=true
        :align: center
        :width: 1510px

| 

* **AutoML at its Heart:** Humans are lazy, at least I am; hence, we want to get things done with Minimal Human Intervention. Sconce was built on this ideology that anyone with nominal knowledge of DL should be able to use this package. Drop your model into the package, call `sconce.compress`, and let it do the magic for you. 
* Compress the model through **Pruning**, **Quantization**, etc. 
* Bring your own dataset and let the **Neural Architecture Search (NAS)** find the best model that fits your deployment constraints. 
* Leverage **Sparsity** in the model and deploy/infer using **Sparse Engines**. 
* Accelerate Inferencing and Reduce Memory Footprint. 
* In addition, this package also supports **Spiking Neural Networks(snnTorch)** in this compression pipeline.

| 

.. image:: https://github.com/satabios/sconce/blob/main/docs/source/images/sconce-overview.jpg?raw=true
        :align: center
        :width: 1510px

|

If you like this project, please consider starring ⭐ this repo as it is the easiest and best way to support it.

Let me know if you are using sconce in any interesting work, research or blogs, as we would love to hear more about it! 

If you have issues, comments, or are looking for advice on training spiking neural networks, you can open an issue, a discussion, 
or chat in our `discord <https://discord.gg/GKwXMrZr>`_ channel.
| 

A Brief workflow is shown below:

.. image:: https://github.com/satabios/sconce/blob/main/docs/source/images/sconce-outline.jpeg?raw=true
        :align: center
        :width: 1510px

| 


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

At present, I am working on adding support for more compression techniques and more models. kindly be patient for feature request/bug fixes. 

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

✌️

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
        :target: https://colab.research.google.com/github/satabios/sconce/blob/main/tutorials/Compression-Pipeline.ipynb#


Here are a few ways you can get started with sconce:


* `Quickstart Notebook (Opens in Colab)`_

* `The API Reference`_ 

* `Tutorials`_

.. _Quickstart Notebook (Opens in Colab): https://colab.research.google.com/github/satabios/sconce/blob/main/tutorials/Compression-Pipeline.ipynb#
.. _The API Reference: https://sconce.readthedocs.io/
.. _Tutorials: https://sconce.readthedocs.io/en/latest/tutorials/index.html

Quickstart:
^^^^^^^^^^^^^^^^^^^^^^^^


Define Network:
^^^^^^^^^^^^^^^^^^^^^^^^


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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

One Roof Solution:
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   sconces.compress()

.. code:: python

    Channel-Wise Pruning (CWP)

   ============================== Comparison Table ==============================
   +---------------------+----------------+--------------+-----------------+
   |                     | Original Model | Pruned Model | Quantized Model |
   +---------------------+----------------+--------------+-----------------+
   | Latency (ms/sample) |      6.4       |     4.7      |       1.9       |
   |     Accuracy (%)    |     93.136     |    90.451    |      90.371     |
   |      Params (M)     |      9.23      |     5.36     |        *        |
   |      Size (MiB)     |     36.949     |    21.484    |      5.419      |
   |       MAC (M)       |      606       |     406      |        *        |
   +---------------------+----------------+--------------+-----------------+


    Granular-Magnitude Pruning(GMP)

    ============================== Comparison Table ==============================
    +---------------------+----------------+--------------+-----------------+
    |                     | Original Model | Pruned Model | Quantized Model |
    +---------------------+----------------+--------------+-----------------+
    | Latency (ms/sample) |      6.2       |     6.3      |       2.2       |
    |     Accuracy (%)    |     93.136     |    92.936    |      92.906     |
    |      Params (M)     |      9.23      |     4.42     |        *        |
    |      Size (MiB)     |     36.949     |    36.949    |      9.293      |
    |       MAC (M)       |      606       |     606      |        *        |
    +---------------------+----------------+--------------+-----------------+


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
