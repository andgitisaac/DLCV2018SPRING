import os
import random
import h5py
import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import rescale
from keras_contrib_modified.keras_contrib.applications.resnet import *
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils import to_categorical


n_classes = 2360
Ntrain, Nval = 56475, 7211
batch_size = 32
base_epochs, finetune_epochs = 300, 150

def Resnet18(input_shape, classes):
    """ResNet with 18 layers and v2 residual units
    """
    return ResNet(input_shape, classes, basic_block, repetitions=[2, 2, 2, 2], include_top=False, dropout=0.3)

def Resnet34(input_shape, classes):
    """ResNet with 34 layers and v2 residual units
    """
    return ResNet(input_shape, classes, basic_block, repetitions=[3, 4, 6, 3], include_top=False, dropout=0.3)

def lr_scheduler(epoch):
    # drops as progression proceeds, good for sgd
    if epoch > 0.9 * base_epochs:
        lr = 0.0001
    elif epoch > 0.75 * base_epochs:
        lr = 0.001
    elif epoch > 0.5 * base_epochs:
        lr = 0.01
    else:
        lr = 0.1
    print('lr: %f' % lr)
    return lr

def finetune_lr_scheduler(epoch):
    # drops as progression proceeds, good for sgd
    if epoch > 0.7 * finetune_epochs:
        lr = 0.0001
    else:
        lr = 0.001
    print('lr: %f' % lr)
    return lr

def batch_gen(dataType, batch_size=8, data_augmentation=None):
    txt_path = "/home/huaijing/DLCV2018SPRING/TA/data/{}_id.txt".format(dataType)
    img_root_path = "/home/huaijing/DLCV2018SPRING/TA/data/{}/".format(dataType)
    dict_path = "/home/huaijing/DLCV2018SPRING/TA/data/seq_labels.npy"
    
    with open(txt_path, 'r') as f:
        raw = [line.rstrip('\n').split(' ') for line in f]
    seq_labels = np.load(dict_path).item()

    Nsample = len(raw)
    while True:
        random_idx = np.arange(Nsample-1)
        np.random.shuffle(random_idx)

        loopTime = Nsample // batch_size
        for i in range(loopTime):
            selected_idx = random_idx[i*batch_size:(i+1)*batch_size]
            filename = [os.path.join(img_root_path, raw[idx][0]) for idx in selected_idx]
            # batch_images = [rescale(imread(f), scale=1.15) for f in filename]
            batch_images = [imread(f) for f in filename]
            batch_labels = [seq_labels[int(raw[idx][1])] for idx in selected_idx]

            batch_images = np.array(batch_images, dtype=np.float32) / 255.0
            batch_labels = to_categorical(np.array(batch_labels, dtype=np.uint16), num_classes=n_classes)
            # print(batch_images.shape)

            if data_augmentation is not None:
                for x_batch, y_batch in data_augmentation.flow(batch_images, batch_labels, batch_size=batch_size, shuffle=False):
                    batch_images = x_batch
                    batch_labels = y_batch
                    break           

            yield batch_images, batch_labels

train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')


# base_model = Resnet34(input_shape=(218, 178, 3), classes=n_classes)
# # base_model = ResNet50(include_top=False, weights=None)

# # content_input = Input(shape=(218, 178, 3), name='content_input')

# # base_model = InceptionV3(input_tensor=content_input, weights='imagenet', include_top=False, pooling='avg')
# x = base_model.output
# x = GlobalAveragePooling2D()(x)

# x = Dropout(0.5)(x)
# predictions = Dense(n_classes, activation='softmax', name='fc1')(x)
# model = Model(inputs=base_model.input, outputs=predictions)



# # for layer in base_model.layers:
# #     layer.trainable = False
# optimizer = SGD(lr=0.1, decay=0.0, momentum=0.9, nesterov=True)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # for i, layer in enumerate(model.layers):
# #    print(i, layer.name)

# chkpt = 'ckpt/drop_augment_resnet34.{epoch:02d}.h5'
# cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
# lr_sch = LearningRateScheduler(lr_scheduler)
# # el_sp = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')
# model.fit_generator(batch_gen('train', batch_size=batch_size, data_augmentation=train_datagen),
#                     epochs=base_epochs,
#                     steps_per_epoch=Ntrain // batch_size,
#                     validation_data=batch_gen('test', batch_size=batch_size),
#                     validation_steps=Nval // batch_size,
#                     verbose=1,
#                     callbacks=[cp_cb, lr_sch])



# for layer in model.layers[:39]:
#    layer.trainable = False
# for layer in model.layers[39:]:
#    layer.trainable = True
model = load_model("/home/huaijing/DLCV2018SPRING/TA/ckpt/drop_augment_resnet34.252.h5")
optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

chkpt = 'ckpt/drop_augment_resnet34_finetune.{epoch:02d}.h5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
# el_sp = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
lr_sch = LearningRateScheduler(finetune_lr_scheduler)
model.fit_generator(batch_gen('train', batch_size=batch_size, data_augmentation=train_datagen),
                    epochs=finetune_epochs,
                    steps_per_epoch=Ntrain // batch_size,
                    validation_data=batch_gen('test', batch_size=batch_size),
                    validation_steps=Nval // batch_size,
                    verbose=1,
                    callbacks=[cp_cb, lr_sch])