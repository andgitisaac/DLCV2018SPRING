import random
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import to_categorical
from utils.reader import getVideoList, readShortVideo



def batch_gen(dir, dataType, batch_size=1):
    if dataType == 'train':
        frames_dicts = []
        labels_dicts = []
        for i in range(1, 5):
            path = dir + 'train{}.npz'.format(i)
            data = np.load(path)
            frames_dicts.append(data['frames'].item())
            labels_dicts.append(data['labels'].item())
            print('TRAIN {} IS LOADED!!!'.format(i))

        while True:
            for frames_dict, labels_dict in zip(frames_dicts, labels_dicts):
                for key, frames in sorted(frames_dict.items(), key=lambda x:random.random()):
                    # loopLimit = len(frames) // batch_size
                    # for i in range(loopLimit):
                    #     frame = frames[i*batch_size:(i+1)*batch_size]
                    #     label = to_categorical([labels_dict[key]]*batch_size, num_classes=11)
                    #     yield frame, label
                    # if len(frames) % batch_size != 0:
                    #     frame = np.zeros((batch_size, 240, 320, 3))
                    #     label = np.zeros((batch_size, 1))
                    #     frame[:len(frames)-batch_size*loopLimit] = frames[batch_size*loopLimit:]
                    #     label[:len(frames)-batch_size*loopLimit] = labels_dict[key]      
                    #     label = to_categorical(label, num_classes=11)
                    #     yield frame, label


                    for frame in sorted(frames, key=lambda x:random.random()):
                        frame = np.expand_dims(frame, axis=0) / 127.5 - 1.0
                        # label = np.expand_dims(to_categorical(labels_dict[key], num_classes=11), axis=0)
                        label = np.expand_dims(labels_dict[key], axis=0)
                        yield frame, label
    else:
        data = np.load(dir+'valid.npz')
        frames_dict = data['frames'].item()
        labels_dict = data['labels'].item()
        while True:
            for key, frames in sorted(frames_dict.items(), key=lambda x:random.random()):
                for frame in sorted(frames, key=lambda x:random.random()):
                    frame = np.expand_dims(frame, axis=0) / 127.5 - 1.0
                    # label = np.expand_dims(to_categorical(labels_dict[key], num_classes=11), axis=0)
                    label = np.expand_dims(labels_dict[key], axis=0)
                    yield frame, label



root_dir = '/home/huaijing/DLCV2018SPRING/hw5/data/'
# batch_size = 2
steps = 29751
validation_steps = 4843


# create the base pre-trained model
content_input = Input(shape=(240, 320, 3))
# base_model = InceptionV3(input_tensor=content_input, weights='imagenet', include_top=False)
base_model = ResNet50(input_tensor=content_input, weights='imagenet', include_top=False, pooling='avg')

x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
predictions = Dense(11, activation='softmax')(x)

model = Model(inputs=content_input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)
# train the model on the new data for a few epochs
chkpt = 'ckpt/Resnet50_base.{epoch:02d}.h5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
# tb = TensorBoard(log_dir='ckpt/base_logs', batch_size=1)
model.fit_generator(batch_gen(root_dir, 'train'),
                    epochs=20,
                    steps_per_epoch=steps,
                    validation_data=batch_gen(root_dir, 'valid'),
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
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
chkpt = 'ckpt/Resnet50_finetune.{epoch:02d}.h5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
# tb = TensorBoard(log_dir='ckpt/finetune_logs', batch_size=1)
model.fit_generator(batch_gen(root_dir, 'train'),
                    epochs=80,
                    steps_per_epoch=steps,
                    validation_data=batch_gen(root_dir, 'valid'),
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[cp_cb])