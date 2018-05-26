import os
import time
import random
import pandas as pd
import skimage
from skimage.io import imread
import numpy as np
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Masking, LSTM, TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop

cnn_path = "/home/huaijing/DLCV2018SPRING/hw5/ckpt/resnetBest_frame.h5"
root_dir = "/home/huaijing/DLCV2018SPRING/hw5/data/FullLengthVideos/"
video_root_dir = os.path.join(root_dir, 'videos')
label_root_dir = os.path.join(root_dir, 'labels')

base_model = load_model(cnn_path)
cnn_input = base_model.get_layer(name='input_1').input
feature_output = base_model.get_layer(name='global_average_pooling2d_1').output
cnn = Model(input=cnn_input, output=feature_output)


def list_files(root):
    r = []    
    dirs = os.listdir(root)
    for dir in sorted(dirs):
        for _, _, files in os.walk(os.path.join(root, dir)):
            lst = []
            for f in sorted(files):
                lst.append(os.path.join(root, dir, f))
            r.append(lst)
    return r

def read_labels(root):
    files = sorted(os.listdir(root))
    lab = []
    for file in files:
        file_path = os.path.join(root, file)
        data = pd.read_csv(file_path, header=None)        
        lab.append(data.ix[:, 0].tolist())
    return lab

def build_model():
    content_input = Input(shape=(None, 2048), batch_shape=(1, None, 2048))
    x = content_input
    x = Masking(mask_value=0.0)(x)
    x = LSTM(units=128, return_sequences=True, dropout=0.5, recurrent_dropout=0.4, recurrent_regularizer=l2(0.01), stateful=True)(x)
    x = LSTM(units=16, return_sequences=True, dropout=0.5, recurrent_dropout=0.4, recurrent_regularizer=l2(0.01), stateful=True)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(11, activation='softmax')(x)
    rnn = Model(inputs=content_input, outputs=x)

    print("loading pretrained rnn weights...")
    rnn_base = load_model('ckpt/trimmed_base.h5')
    for new_layer, layer in zip(rnn.layers[1:], rnn_base.layers[1:]):
        new_layer.set_weights(layer.get_weights())
    print("Weights is set!")

    rnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    rnn.summary()
    return rnn


epochs = 80
max_length = 400
train_label_list = read_labels(os.path.join(label_root_dir, 'train'))
train_file_list = list_files(os.path.join(video_root_dir, 'train'))
valid_label_list = read_labels(os.path.join(label_root_dir, 'valid'))
valid_file_list = list_files(os.path.join(video_root_dir, 'valid'))

rnn = build_model()

# Count number of steps in a epoch
global_step = 0
for video_sublist in train_file_list:
    video_length = len(video_sublist)
    loopTime = video_length // max_length
    for steps in range(loopTime+1):
        global_step += 1

max_val_acc = 0
for ep in range(epochs):
    start_time = time.time()
    totalLoss, totalAcc, step_count = 0, 0, 0

    # Training
    combine = list(zip(train_file_list, train_label_list))
    random.shuffle(combine)
    for video_sublist, label_sublist in combine:
        video_length = len(video_sublist)
        loopTime = video_length // max_length

        # Start training on a whole videos
        for steps in range(loopTime+1):
            frames = np.zeros((max_length, 240, 320, 3), dtype=np.uint8)
            labels = np.zeros((max_length,), dtype=np.uint8)
            iter_step = (video_length-steps*max_length) if (steps == loopTime) else max_length

            for i in range(iter_step):
                f = video_sublist[steps*max_length+i]            
                frames[i] = imread(f)
                labels[i] = label_sublist[steps*max_length+i]
            
            features = cnn.predict(frames, batch_size=16)
            features = np.expand_dims(features, axis=0)
            labels = np.expand_dims(to_categorical(labels, num_classes=11), axis=0)
            
            loss, acc = rnn.train_on_batch(features, labels)
            totalLoss += loss
            totalAcc += acc
            step_count += 1
            print("Epoch: {}/{} Step: {}/{} Loss: {:.4f} Acc: {:.4f}".format(ep+1, epochs, step_count, global_step, totalLoss/(step_count + 1), totalAcc/(step_count + 1)))
        
        rnn.reset_states()
    print('{:.2f} secs/epoch'.format(time.time()-start_time))    

    # Testing
    total_count, total_video_length = 0, 0
    for video_sublist, label_sublist in zip(valid_file_list, valid_label_list):
        video_length = len(video_sublist)
        total_video_length += video_length
        loopTime = video_length // max_length        
        whole_pred_label = np.zeros(((loopTime+1)*max_length,), dtype=np.uint8)

        for steps in range(loopTime+1):
            frames = np.zeros((max_length, 240, 320, 3), dtype=np.uint8)
            labels = np.zeros((max_length,), dtype=np.uint8)
            iter_step = (video_length-steps*max_length) if (steps == loopTime) else max_length

            for i in range(iter_step):
                f = video_sublist[steps*max_length+i]            
                frames[i] = imread(f)
                labels[i] = label_sublist[steps*max_length+i]

            features = cnn.predict(frames, batch_size=16)
            features = np.expand_dims(features, axis=0)

            batch_pred_label = rnn.predict(features, batch_size=16)
            batch_pred_label = np.argmax(batch_pred_label, axis=-1)
            whole_pred_label[steps*max_length:(steps+1)*max_length] = batch_pred_label[0, :]
        for target, pred in zip(label_sublist, whole_pred_label[:video_length]):
            if target == pred:
                total_count += 1
    val_acc = total_count/total_video_length
    print("Validation Dataset Acc: {:.5f}".format(val_acc))

    if val_acc > max_val_acc:
        model_name = 'ckpt/full.{:02d}.h5'.format(ep+1)
        print('val_acc improves from {:.6f} to {:.6f}, saving model to {}'.format(max_val_acc, val_acc, model_name))
        rnn.save(model_name)
        max_val_acc = val_acc
    else:
        print('val_acc does not improve.')

            





    

        

            

        



