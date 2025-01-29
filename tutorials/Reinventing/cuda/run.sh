#!/bin/bash

read -p "Enter File Name: " file_name
nvcc -arch=sm_80 $file_name.cu -o $file_name
echo "----------Output----------"
./$file_name