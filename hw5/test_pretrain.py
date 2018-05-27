import numpy as np
import h5py
from keras.models import Model, load_model

root_dir = '/home/huaijing/DLCV2018SPRING/hw5/data/'
resnet_path = 'model/p1/resnet_base.h5'
dense_path = "model/p1/dense.acc5300.h5"

def read_frames(dir, dataType):
    print("Loading {} dataset".format(dataType))
    path = dir + '{}.h5'.format(dataType)
    with h5py.File(path, 'r') as hf:
        frames = hf['frames'][:]
        labels = hf['labels'][:]
        start = hf['start_idx'][:]
        end = hf['end_idx'][:]
    print("{} dataset is loaded!".format(dataType))
    return frames, labels, start, end

def base_model_predict(base_model, frames, length):
    feature_concat = base_model.predict(frames, batch_size=4)
    feature_input = np.mean(feature_concat, axis=0)
    feature_input = np.expand_dims(feature_input, axis=0)
    return feature_input


print('Loading resnet model...')
resnet = load_model(resnet_path)
resnet_input = resnet.get_layer(name='input_1').input
feature_output = resnet.get_layer(name='global_average_pooling2d_1').output
base_model = Model(input=resnet_input, output=feature_output)

print('Loading dense model...')
dense = load_model(dense_path)

valid_frames, valid_labels, valid_start, valid_end = read_frames(root_dir, 'valid')
Nvalid = valid_start.shape[0]

print("Evaluating Validation Dataset...")
count = 0
with open('p1_valid.txt','w') as file:
    for idx in range(Nvalid):
        start_idx, end_idx = valid_start[idx], valid_end[idx]
        target_frames = valid_frames[start_idx:end_idx]
        target_labels = np.expand_dims(valid_labels[idx], axis=0)
        feature_input = base_model_predict(base_model, target_frames, end_idx-start_idx)
        pred_label = np.argmax(dense.predict(feature_input))

        print('Predicting {}/{}: pred: {}, gt: {}'.format(idx+1, Nvalid, pred_label, valid_labels[idx]))
        file.write(str(pred_label))
        file.write('\n')

        if pred_label == target_labels:
            count += 1


val_acc = count / Nvalid
print("Validation Acc: {:.4f}".format(val_acc))



