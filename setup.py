
from setuptools import setup, find_packages
import os
os.environ['PIP_DEFAULT_TIMEOUT'] = '100'


# with open('README.rst') as readme_file:
#     README = readme_file.read()
# import os
# setup_args = dict(
#     name='sconce',
#     version='0.0.70',
#     description='sconce: torch helper',
#     long_description_content_type="text/markdown",
#     long_description=README + '\n\n',
#     packages=['sconce'],
#     author='Sathyaprakash Narayanan',
#     author_email='snaray17@ucsc.edu',
#     url='https://github.com/satabios/sconce',
#     download_url='https://pypi.org/project/sconce/'
# )

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []

if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()



from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

# setup(**setup_args, setup_requires=install_requires, install_requires=install_requires)


# fmt: off
__version__ = '0.57'
# fmt: on


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

    install_requires=install_requires,
    license="MIT License",
    long_description=readme,
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