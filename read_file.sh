#!/usr/bin/bash
filename="pyproject.toml"
while read -r line; do
    name="$line"
    echo "$name"
done < "$filename"