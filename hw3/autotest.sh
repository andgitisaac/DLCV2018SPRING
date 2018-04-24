#!/bin/bash

# Train an autoencoder with depth 5 to 1, and 
# saving their reconstruted images respectively.
for i in $(seq 30 60)
do
	python3 -u test.py --ep-h5 ${i} --net segnet
    python3 mean_iou_evaluate.py -p output/ -g data/validation/

done
exit 0

