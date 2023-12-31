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


Quantization Techniques