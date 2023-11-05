"""The setup script."""

from setuptools import setup, find_packages

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

setup(
    author="Sathyaprakash Narayanan",
    author_email="snaray17@ucsc.edu",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 0.57 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    description="Model Compresion Made Easy",
    long_description=readme,
    install_requires=install_requires,
    license="MIT License",
    include_package_data=True,
    keywords="sconce",
    name="sconce",
    packages=find_packages(include=["sconce"]),
    # test_suite="tests",
    # tests_require=test_requirements,
    url="https://github.com/satabios/sconce",
    version=__version__,
    zip_safe=False,
)

