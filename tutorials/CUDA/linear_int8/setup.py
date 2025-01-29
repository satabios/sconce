from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='linear_w8a8',
    ext_modules=[
        CUDAExtension('linear_w8a8', [
            'linear_w8a8.cpp',
            'linear_w8a8_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

