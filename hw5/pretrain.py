import time
import random
import h5py
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
from utils.reader import getVideoList, readShortVideo


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
    # feature_concat = np.empty((length, 2048), dtype=np.float32)
    # for i, frame in enumerate(frames):
    #     frame = np.expand_dims(frame, axis=0)
    #     feature_concat[i, :] = base_model.predict(frame)
    feature_concat = base_model.predict(frames, batch_size=4)
    feature_input = np.mean(feature_concat, axis=0)
    feature_input = np.expand_dims(feature_input, axis=0)
    return feature_input


root_dir = '/home/huaijing/DLCV2018SPRING/hw5/data/'
epochs = 80
batch_size = 4


train_frames, train_labels, train_start, train_end = read_frames(root_dir, 'train')
valid_frames, valid_labels, valid_start, valid_end = read_frames(root_dir, 'valid')
Ntrain, Nvalid = train_start.shape[0], valid_start.shape[0]

# create the base pre-trained model
# content_input = Input(shape=(240, 320, 3))
resnet = load_model('ckpt/resnetBest_frame.h5')

resnet_input = resnet.get_layer(name='input_1').input
feature_output = resnet.get_layer(name='global_average_pooling2d_1').output
base_model = Model(input=resnet_input, output=feature_output)
# base_model.summary()


# base_model = ResNet50(input_tensor=content_input, weights='imagenet', include_top=False, pooling='avg')

feature_input = Input(shape=(2048,))
predictions = Dense(11, activation='softmax')(feature_input)
model = Model(inputs=feature_input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

max_val_acc = 0
index = np.arange(Ntrain)
for ep in range(epochs):
    start_time = time.time()
    totalLoss, totalAcc = 0, 0
    np.random.shuffle(index)

    steps = 3236 // batch_size
    for step in range(steps):
        idx = index[step*batch_size:(step+1)*batch_size]
        start_idx, end_idx = train_start[idx], train_end[idx]
        target_labels = train_labels[idx]
        feature_input = np.empty((batch_size, 2048), dtype=np.float32)
        for i in range(batch_size):
            target_frames = train_frames[start_idx[i]:end_idx[i]]
            feature_input[i] = base_model_predict(base_model, target_frames, end_idx[i]-start_idx[i])
        loss, acc = model.train_on_batch(feature_input, target_labels)
        totalLoss += loss
        totalAcc += acc
        print("Epoch: {}/{} Step: {}/{} Loss: {:.4f} Acc: {:.4f}".format(ep+1, epochs, step+1, steps, totalLoss/(step + 1), totalAcc/(step + 1)))



    # for step, idx in enumerate(index):
        
    #     start_idx, end_idx = train_start[idx], train_end[idx]
    #     target_frames = train_frames[start_idx:end_idx]
    #     target_labels = np.expand_dims(train_labels[idx], axis=0)
    #     feature_input = base_model_predict(base_model, target_frames, end_idx-start_idx)        
        
    #     loss, acc = model.train_on_batch(feature_input, target_labels)
    #     totalLoss += loss
    #     totalAcc += acc
    #     print("Epoch: {}/{} Step: {}/{} Loss: {:.4f} Acc: {:.4f}".format(ep+1, epochs, step+1, Ntrain, totalLoss/(step + 1), totalAcc/(step + 1)))
    
    print('{:.2f} secs/epoch'.format(time.time()-start_time))
    print("Evaluating Validation Dataset...")
    count = 0
    for idx in range(Nvalid):
        start_idx, end_idx = valid_start[idx], valid_end[idx]
        target_frames = valid_frames[start_idx:end_idx]
        target_labels = np.expand_dims(valid_labels[idx], axis=0)
        feature_input = base_model_predict(base_model, target_frames, end_idx-start_idx)
        pred_label = np.argmax(model.predict(feature_input))
        if pred_label == target_labels:
            count += 1
    val_acc = count / Nvalid
    print("Validation Acc: {:.4f}".format(val_acc))

    if val_acc > max_val_acc:
        model_name = 'ckpt/resnet50_finetune.{:02d}.h5'.format(ep+1)
        print('val_acc improves from {:.6f} to {:.6f}, saving model to {}'.format(max_val_acc, val_acc, model_name))
        model.save(model_name)
        max_val_acc = val_acc
    else:
        print('val_acc does not improve.')









