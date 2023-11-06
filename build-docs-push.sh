#! /bin/bash



git_push() {

git add .
git commit -m "$1"
git push -u origin master
}


cd docs/
make html
cd ../

# echo "Message to Push?"
# read message
# git_push "$message"



