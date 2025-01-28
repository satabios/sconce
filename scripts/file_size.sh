#! /bin/bash
#Find the layer files
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  awk '{size=$2/1024/1024; printf "%s %.2f MB %s\n", $1, size, $3}' | \
  sort -k2nr | \
  head -10