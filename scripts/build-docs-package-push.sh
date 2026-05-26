#! /bin/bash

## add files, commit and push function for git
git_push() {

git add .
git commit -m "$1"
git push -u origin main
}


#Remove all build files
rm -rf dist/
rm -rf build/
rm -rf .eggs/

#Run Tests
uv run flake8 sconce scripts
uv run black sconce scripts

echo "Version-To-Update?"
read value
uv version "$value"




#Generate rst for tutorials
#Generate rst files
#tutorials_folder_path="tutorials"
#for file in "$tutorials_folder_path"/*.ipynb; do
#  jupyter nbconvert --to rst --output-dir="docs/source/tutorials/" "$file"
#done


#Build Docs
uv run sphinx-build -b html docs/source docs/_build/html

#Build Package
uv build
# twine upload dist/* --verbose

#Push to GitHub




echo "Message to Push?"
read message
git_push "$message"
