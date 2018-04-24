#!/bin/bash
fileid="14O4F9tNvFWd_U5RL8cpsW41Edb5dJkVv"
filename="vgg16_weights.h5"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
