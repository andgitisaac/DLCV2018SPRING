import random
import h5py
import numpy as np
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Dropout, Masking, LSTM, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
from utils.reader import getVideoList, readShortVideo


def batch_gen(dir, dataType, batch_size=4):
    path = dir + '{}.h5'.format(dataType)
    with h5py.File(path, 'r') as hf:
        frames = hf['frames'][:]
        labels = hf['labels'][:]
        start = hf['start_idx'][:]
        end = hf['end_idx'][:]
    
    global_steps = labels.shape[0]

    if dataType == 'train':
        random_idx = np.arange(global_steps-1)
        loopTime = global_steps // batch_size

        while True:        
            np.random.shuffle(random_idx)
            for step in range(loopTime):
                idx = random_idx[step*batch_size:(step+1)*batch_size]
                start_idx, end_idx = start[idx], end[idx]
                batch_labels = labels[idx]

                max_frame_length = 0
                for s, e in zip(start_idx, end_idx):
                    max_frame_length = max(max_frame_length, e-s)                
                batch_frames = np.zeros((batch_size, max_frame_length, 240, 320, 3), dtype=np.uint8)

                for i, (s, e) in enumerate(zip(start_idx, end_idx)):
                    batch_frames[i, :e-s, :, :, :] = frames[s:e, :, :, :]            
                yield batch_frames, batch_labels
    elif dataType == 'valid':
        while True:
            for idx, (target_label, s, e) in enumerate(zip(labels, start, end)):
                batch_frames = np.expand_dims(frames[s:e, :, :, :], axis=0)
                target_label = np.expand_dims(target_label, axis=0)
                yield batch_frames, target_label

def build_model():
    hidden_unit = 512
    content_input = Input(shape=(None, 240, 320, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    resnet = load_model('ckpt/resnetBest_frame.h5')
    for new_layer, layer in zip(base_model.layers[1:], resnet.layers[1:]):
        new_layer.set_weights(layer.get_weights())

    
    # content_input = resnet.get_layer(name='input_1').input
    # feature_output = resnet.get_layer(name='global_average_pooling2d_1').output
    # base_model = Model(input=content_input, output=feature_output)

    base_model.trainable = False
    feature_output = TimeDistributed(base_model)(content_input)
    x = feature_output
    x = Masking(mask_value=0.0)(x)
    x = LSTM(units=hidden_unit, return_sequences=False, dropout=0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(11, activation='softmax')(x)


    return Model(inputs=content_input, outputs=x)
    


root_dir = '/home/huaijing/DLCV2018SPRING/hw5/data/'
batch_size = 1
steps = 10 // batch_size
validation_steps = 517

model = build_model()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

chkpt = 'ckpt/trimmed_train.{epoch:02d}.h5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, cooldown=5, min_lr=1e-4, verbose=1)
csv_logger = CSVLogger('trim_train.log')
model.fit_generator(batch_gen(root_dir, 'valid', batch_size),
                    epochs=100,
                    steps_per_epoch=steps,
                    validation_data=batch_gen(root_dir, 'valid', batch_size),
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[cp_cb, csv_logger])
