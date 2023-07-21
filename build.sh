#! /bin/bash


rm ../dist/* -y
python3 setup.py sdist bdist_wheel
twine upload dist/* --verbose