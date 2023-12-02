#! /bin/bash

## add files, commit and push function for git
git_push() {

git add .
git commit -m "$1"
git push -u origin main
}

git pull
#Remove all build files
rm -rf dist/
rm -rf build/
rm -rf .eggs/

#Run Tests
flake8 sconce test
black sconce test

echo "Version-To-Update?"
read value
version="version = \"${value}\""
sed -i "7s/.*/$version/" pyproject.toml


#Generate rst for tutorials
#Generate rst files
tutorials_folder_path="tutorials"
for file in "$tutorials_folder_path"/*.ipynb; do
  jupyter nbconvert --to rst --output-dir="docs/source/tutorials/" "$file"
done


#Build Docs
cd docs/
make html
cd ../

#Build Package
python3 setup.py clean --all sdist bdist_wheel
twine upload dist/* --verbose

#Push to GitHub




echo "Message to Push?"
read message
git_push "$message"