import numpy as np
import h5py
from keras.models import load_model

model_path = 'ckpt/Resnet50_finetune.{:02d}.h5'.format(10)
data_path = '/home/huaijing/DLCV2018SPRING/hw5/data/valid.h5'

model = load_model(model_path)
with h5py.File(data_path, 'r') as hf:
    frames = hf['frames'][:]
    labels = hf['labels'][:]    

length = labels.shape[0]
count = 0
for idx, (frame, label) in enumerate(zip(frames, labels)):    
    pred_label = model.predict(np.expand_dims(frame, axis=0))
    pred_label = np.argmax(pred_label)
    print('Predicting {}/{}: pred: {}, gt: {}'.format(idx+1, length, pred_label, label))
    if pred_label == label:
        count += 1
print("Accuracy of {}: {}".format(model_path, count/length*100))

