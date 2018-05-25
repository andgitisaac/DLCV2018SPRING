import random
import h5py
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
from utils.reader import getVideoList, readShortVideo


def batch_gen(dir, dataType, batch_size=4):
    path = dir + '{}.h5'.format(dataType)
    with h5py.File(path, 'r') as hf:
        frames = hf['frames'][:]
        labels = hf['labels'][:]
        indexes = hf['indexes'][:]
    while True:
        length = indexes.shape[0]
        random_idx = np.arange(length-1)
        np.random.shuffle(random_idx)

        loopTime = length // batch_size
        for i in range(loopTime):
            batch_frames = frames[random_idx[i*batch_size:(i+1)*batch_size]]
            batch_labels = labels[random_idx[i*batch_size:(i+1)*batch_size]]
            yield batch_frames, batch_labels


root_dir = '/home/huaijing/DLCV2018SPRING/hw5/data/'
batch_size = 8
steps = 29751 // batch_size
validation_steps = 4843 // batch_size

# create the base pre-trained model
content_input = Input(shape=(240, 320, 3))
base_model = ResNet50(input_tensor=content_input, weights='imagenet', include_top=False, pooling='avg')

x = base_model.output
predictions = Dense(11, activation='softmax')(x)

model = Model(inputs=content_input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# train the model on the new data for a few epochs
chkpt = 'ckpt/Resnet50_base.{epoch:02d}.h5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.fit_generator(batch_gen(root_dir, 'train', batch_size),
                    epochs=20,
                    steps_per_epoch=steps,
                    validation_data=batch_gen(root_dir, 'valid', batch_size),
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[cp_cb])

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:141]:
   layer.trainable = False
for layer in model.layers[141:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
# optimizer = Adam(lr=1e-4)
optimizer = RMSprop(lr=1e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
chkpt = 'ckpt/Resnet50_finetune.{epoch:02d}.h5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.fit_generator(batch_gen(root_dir, 'train', batch_size),
                    epochs=80,
                    steps_per_epoch=steps,
                    validation_data=batch_gen(root_dir, 'valid', batch_size),
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[cp_cb])