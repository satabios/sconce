
from setuptools import setup, find_packages
import os
os.environ['PIP_DEFAULT_TIMEOUT'] = '100'


with open('README.md') as readme_file:
    README = readme_file.read()
import os
setup_args = dict(
    name='sconce',
    version='0.0.40',
    description='sconce: torch helper',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n',
    packages=['sconce'],
    author='Sathyaprakash Narayanan',
    author_email='snaray17@ucsc.edu',
    url='https://github.com/satabios/sconce',
    download_url='https://pypi.org/project/sconce/'
)

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []

if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(**setup_args, setup_requires=install_requires, install_requires=install_requires)
