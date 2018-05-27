import numpy as np
import h5py
from keras.models import Model, load_model

model_path = "/home/huaijing/DLCV2018SPRING/hw5/model/p2/trimmed_acc5261.h5"
data_path = '/home/huaijing/DLCV2018SPRING/hw5/data/train_features.h5'

model = load_model(model_path)
input = model.get_layer(name='input_1').input
rnn_output = model.get_layer(name='lstm_2').output
rnn = Model(input=input, output=rnn_output)

with h5py.File(data_path, 'r') as hf:
    features = hf['features'][:]
    labels = hf['labels'][:]
    start = hf['start_idx'][:]
    end = hf['end_idx'][:] 

length = labels.shape[0]
rnn_features = np.empty((length, 16), dtype=np.float32)

with h5py.File('data/rnn_feature_train.h5', 'w') as hf:
    for idx, (target_label, s, e) in enumerate(zip(labels, start, end)):
        feature = np.expand_dims(features[s:e, :], axis=0)
        rnn_feature = rnn.predict(feature)
        rnn_features[idx] = rnn_feature
    hf.create_dataset("features",  data=rnn_features)
