Usage
=====

.. _installation:

Installation
------------

To use sconce, first install it using pip:

.. code-block:: console

   (.venv) $ pip install sconce

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``sconce.compress()`` function:

.. autofunction:: sconce.compress

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`sconce.compress`
will raise an exception.


For example:

>>> import sconce
>>> sconces = sconce
>>> sconces.compress()
['shells', 'gorgonzola', 'parsley']

