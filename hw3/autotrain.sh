#!/bin/bash

python3 -u train.py --net fcn32
python3 -u train.py --net fcn8
python3 -u train.py --net segnet
exit 0

