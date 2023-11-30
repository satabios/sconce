#! /bin/bash



git_push() {

git add .
git commit -m "$1"
git push -u origin main
}



#Generate rst files
tutorials_folder_path="tutorials"
for file in "$tutorials_folder_path"/*.ipynb; do
  jupyter nbconvert --to rst --output-dir="docs/source/tutorials/" "$file"
done



cd docs/
make html
cd ../

echo "Message to Push?"
read message
git_push "$message"


