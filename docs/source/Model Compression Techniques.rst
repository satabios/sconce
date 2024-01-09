=================================
Model Compression Techniques
=================================


Pruning Techniques
###################


.. list-table:: Method Comparison
   :widths: 25 15 15 25 15 15 25 25
   :header-rows: 1

   * - Method
     - Weight Update
     - Calibration Data
     - Pruning Metric
     - Complexity
     - Fine-Tuning
     - Support for LLM/Attention/Linear
     - Support for Convolutional Layer
   * - Granular-Magnitude
     - ✗
     - **✗**
     - :math:`|Wij|`
     - O(1)
     - ✓
     - ✓
     - ✓
   * - Channel-Wise Magnitude
     - ✗
     - ✓
     - :math:`|Wj|`
     - O(1)
     - ✓
     - ✓
     - ✓
   * - Optimal Brain Compression
     - ✗
     - ✓
     - :math:`|W|^2/diag(XXT + λI)−1`
     - O(d^3 hidden)
     - ✗
     - ✓
     - ✓
   * - SparseGPT
     - ✓
     - ✓
     - :math:`|W|^2/diag(XXT + λI)−1`
     - O(d^3 hidden)
     - ✗
     - ✓
     - ✗
   * - Wanda
     - ✗
     - ✓
     - :math:`|W_{ij}|. |X_{j}|_{2}`
     - O(d^2 hidden)
     - ✗
     - ✓
     - ✗
   * - **Venum**
     - **✗**
     - ✓
     - :math:`|W_{ij}|. |X_{j}|_{2}`
     - O(d^2 hidden)
     - Minimal(optional)
     - ✓
     - ✓



Quantization Techniques
########################

Currently, the package only suports Eager Mode Quantization. I look forward to integerate FX Graph Mode Quantization in the near future.

There are three types of quantization supported:

1. Dynamic Quantization:
   - Weights are quantized with activations read/stored in floating point and quantized for compute.

2. Static Quantization:
   - Weights are quantized.
   - Activations are quantized.
   - Calibration is required post-training.

3. Static Quantization Aware Training:
   - Weights are quantized.
   - Activations are quantized.
   - Quantization numerics are modeled during training.
