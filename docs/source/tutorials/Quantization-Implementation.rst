Create a simple tensor with random items
========================================

.. code:: ipython3

    import numpy as np
    
    # Suppress scientific notation
    np.set_printoptions(suppress=True)
    
    # Generate randomly distributed parameters
    params = np.random.uniform(low=-50, high=150, size=20)
    
    # Make sure important values are at the beginning for better debugging
    params[0] = params.max() + 1
    params[1] = params.min() - 1
    params[2] = 0
    
    # Round each number to the second decimal place
    params = np.round(params, 2)
    
    # Print the parameters
    print(params)


.. parsed-literal::

    [127.48 -40.1    0.    89.74 124.38 -39.1  126.48  21.2  -35.99 124.16
       5.92  41.68  23.6  -26.4  -21.51 -20.6   94.49  85.07  70.11  76.91]


Define the quantization methods and quantize
============================================

.. code:: ipython3

    def clamp(params_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
        params_q[params_q < lower_bound] = lower_bound
        params_q[params_q > upper_bound] = upper_bound
        return params_q
    
    def asymmetric_quantization(params: np.array, bits: int) -> tuple[np.array, float, int]:
        # Calculate the scale and zero point
        alpha = np.max(params)
        beta = np.min(params)
        scale = (alpha - beta) / (2**bits-1)
        zero = -1*np.round(beta / scale)
        lower_bound, upper_bound = 0, 2**bits-1
        # Quantize the parameters
        quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
        return quantized, scale, zero
    
    def asymmetric_dequantize(params_q: np.array, scale: float, zero: int) -> np.array:
        return (params_q - zero) * scale
    
    def symmetric_dequantize(params_q: np.array, scale: float) -> np.array:
        return params_q * scale
    
    def symmetric_quantization(params: np.array, bits: int) -> tuple[np.array, float]:
        # Calculate the scale
        alpha = np.max(np.abs(params))
        scale = alpha / (2**(bits-1)-1)
        lower_bound = -2**(bits-1)
        upper_bound = 2**(bits-1)-1
        # Quantize the parameters
        quantized = clamp(np.round(params / scale), lower_bound, upper_bound).astype(np.int32)
        return quantized, scale
    
    def quantization_error(params: np.array, params_q: np.array):
        # calculate the MSE
        return np.mean((params - params_q)**2)
    
    (asymmetric_q, asymmetric_scale, asymmetric_zero) = asymmetric_quantization(params, 8)
    (symmetric_q, symmetric_scale) = symmetric_quantization(params, 8)
    
    print(f'Original:')
    print(np.round(params, 2))
    print('')
    print(f'Asymmetric scale: {asymmetric_scale}, zero: {asymmetric_zero}')
    print(asymmetric_q)
    print('')
    print(f'Symmetric scale: {symmetric_scale}')
    print(symmetric_q)


.. parsed-literal::

    Original:
    [127.48 -40.1    0.    89.74 124.38 -39.1  126.48  21.2  -35.99 124.16
       5.92  41.68  23.6  -26.4  -21.51 -20.6   94.49  85.07  70.11  76.91]
    
    Asymmetric s: 0.6571764705882354, z: 61.0
    [255   0  61 198 250   2 253  93   6 250  70 124  97  21  28  30 205 190
     168 178]
    
    Symmetric s: 1.003779527559055
    [127 -40   0  89 124 -39 126  21 -36 124   6  42  24 -26 -21 -21  94  85
      70  77]


.. code:: ipython3

    # Dequantize the parameters back to 32 bits
    params_deq_asymmetric = asymmetric_dequantize(asymmetric_q, asymmetric_scale, asymmetric_zero)
    params_deq_symmetric = symmetric_dequantize(symmetric_q, symmetric_scale)
    
    print(f'Original:')
    print(np.round(params, 2))
    print('')
    print(f'Dequantize Asymmetric:')
    print(np.round(params_deq_asymmetric,2))
    print('')
    print(f'Dequantize Symmetric:')
    print(np.round(params_deq_symmetric, 2))


.. parsed-literal::

    Original:
    [127.48 -40.1    0.    89.74 124.38 -39.1  126.48  21.2  -35.99 124.16
       5.92  41.68  23.6  -26.4  -21.51 -20.6   94.49  85.07  70.11  76.91]
    
    Dequantize Asymmetric:
    [127.49 -40.09   0.    90.03 124.21 -38.77 126.18  21.03 -36.14 124.21
       5.91  41.4   23.66 -26.29 -21.69 -20.37  94.63  84.78  70.32  76.89]
    
    Dequantize Symmetric:
    [127.48 -40.15   0.    89.34 124.47 -39.15 126.48  21.08 -36.14 124.47
       6.02  42.16  24.09 -26.1  -21.08 -21.08  94.36  85.32  70.26  77.29]


.. code:: ipython3

    # Calculate the quantization error
    print(f'{"Asymmetric error: ":>20}{np.round(quantization_error(params, params_deq_asymmetric), 2)}')
    print(f'{"Symmetric error: ":>20}{np.round(quantization_error(params, params_deq_symmetric), 2)}')


.. parsed-literal::

      Asymmetric error: 0.03
       Symmetric error: 0.08


