#!/bin/bash
python3 test.py --net fcn32 --model-path VGG_FCN32.h5 --input-path $1 --output-path $2
