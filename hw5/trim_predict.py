import numpy as np
import h5py
from keras.models import load_model

model_path = "/home/huaijing/DLCV2018SPRING/hw5/model/p2/trimmed_acc5261.h5"
data_path = '/home/huaijing/DLCV2018SPRING/hw5/data/valid_features.h5'

model = load_model(model_path)
model.summary()
with h5py.File(data_path, 'r') as hf:
    features = hf['features'][:]
    labels = hf['labels'][:]
    start = hf['start_idx'][:]
    end = hf['end_idx'][:] 

length = labels.shape[0]
count = 0
for idx, (target_label, s, e) in enumerate(zip(labels, start, end)):
    feature = np.expand_dims(features[s:e, :], axis=0)
    pred_label = model.predict(feature)
    pred_label = np.argmax(pred_label)
    print('Predicting {}/{}: pred: {}, gt: {}'.format(idx+1, length, pred_label, target_label))
    if pred_label == target_label:
        count += 1
print("Accuracy of {}: {}".format(model_path, count/length*100))

