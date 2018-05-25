import random
import h5py
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Masking, LSTM, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
from utils.reader import getVideoList, readShortVideo


def batch_gen(dir, dataType, batch_size=4):
    path = dir + '{}_features.h5'.format(dataType)
    with h5py.File(path, 'r') as hf:
        features = hf['features'][:]
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
                batch_features = np.zeros((batch_size, max_frame_length, 2048), dtype=np.float32)

                for i, (s, e) in enumerate(zip(start_idx, end_idx)):
                    batch_features[i, :e-s, :] = features[s:e, :]            
                yield batch_features, batch_labels
    elif dataType == 'valid':
        while True:
            for idx, (target_label, s, e) in enumerate(zip(labels, start, end)):
                feature = np.expand_dims(features[s:e, :], axis=0)
                target_label = np.expand_dims(target_label, axis=0)
                yield feature, target_label

def build_model():
    hidden_unit = 512
    content_input = Input(shape=(None, 2048))
    x = content_input
    x = Masking(mask_value=0.0)(content_input)
    x = LSTM(units=hidden_unit, return_sequences=False, dropout=0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(11, activation='softmax')(x)
    return Model(inputs=content_input, outputs=x)
    


root_dir = '/home/huaijing/DLCV2018SPRING/hw5/data/'
batch_size = 8
steps = 3236 // batch_size
validation_steps = 517

model = build_model()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# path = root_dir + '{}_features.h5'.format('train')
# with h5py.File(path, 'r') as hf:
#     features = hf['features'][:]
#     labels = hf['labels'][:]
#     start = hf['start_idx'][:]
#     end = hf['end_idx'][:]

# global_steps = labels.shape[0]
# random_idx = np.arange(global_steps-1)
# loopTime = global_steps // batch_size


# np.random.shuffle(random_idx)
# for step in range(10):
#     idx = random_idx[step*batch_size:(step+1)*batch_size]
#     start_idx, end_idx = start[idx], end[idx]
#     batch_labels = labels[idx]

#     max_frame_length = 0
#     for s, e in zip(start_idx, end_idx):
#         max_frame_length = max(max_frame_length, e-s)
    
#     # batch_features = np.zeros((batch_size, max_frame_length, 2048), dtype=np.float32)
#     batch_features = np.ones((batch_size, max_frame_length, 2048), dtype=np.float32) * 255.0
#     for i, (s, e) in enumerate(zip(start_idx, end_idx)):
#         batch_features[i, :e-s, :] = features[s:e, :]   
    
#     predict_label = model.predict(batch_features)
#     get_mask_output = K.function([model.layers[0].input], [model.layers[1].output])
#     mask_output = get_mask_output([batch_features])[0]
#     # print(batch_features[:, :, 3])
#     print(len(get_mask_output([batch_features])))
#     print(mask_output[:, :, :3], mask_output.shape)
#     for lab, s, e in zip(predict_label, start_idx, end_idx):
#         print("predict: {}, length: {}".format(lab, e-s))




chkpt = 'ckpt/trimmed_train.{epoch:02d}.h5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, cooldown=5, min_lr=1e-4, verbose=1)
csv_logger = CSVLogger('trim_train.log')
model.fit_generator(batch_gen(root_dir, 'train', batch_size),
                    epochs=100,
                    steps_per_epoch=steps,
                    validation_data=batch_gen(root_dir, 'valid', batch_size),
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[cp_cb, csv_logger])
