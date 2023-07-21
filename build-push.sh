
git_push() {

git add .
git commit -m "$1"
git push -u origin master
}


rm -rf ../dist/*
rm -rf ../build/* 
rm -rf ../.eggs/*
python3 setup.py sdist bdist_wheel

echo "Message to Push?"
read message
git_push "$message"



