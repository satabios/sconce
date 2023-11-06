#! /bin/bash

git_push() {

git add .
git commit -m "$1"
git push -u origin master
}


rm -rf dist/
rm -rf build/
rm -rf .eggs/



echo "Version-To-Update?"
read value
version="version = \"${value}\""
sed -i "3s/.*/$version/" pyproject.toml


cd docs/
make html
cd ../



python3 setup.py clean --all sdist bdist_wheel
twine upload dist/* --verbose

echo "Message to Push?"
read message
git_push "$message"
