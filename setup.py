from setuptools import setup, find_packages

import os


with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

# fmt: off
__version__ = '0.57'
# fmt: on

requirements = [
    "torch>=1.1.0",
    "pandas",
    "matplotlib",
    "numpy>=1.17",
    "nir",
    "nirtorch",
]


test_requirements = ["pytest>=6"]

version = __version__

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requirements = []

if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requirements = f.read().splitlines()

setup(
    author="Sathyaprakash Narayanan",
    author_email="snaray17@ucsc.edu",
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    description="Model Compresion Made Easy",
    long_description=readme,
    install_requires=install_requirements,
    license="MIT License",
    include_package_data=True,
    keywords="sconce",
    name="sconce",
    packages=find_packages(include=["sconce"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/satabios/sconce",
    version=__version__,
    zip_safe=False,
)

