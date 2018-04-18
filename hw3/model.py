from __future__ import print_function, division

import numpy as np
from ops import batch_gen, binary_crossentropy_with_logits
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from utils import BilinearUpSampling2D


class VGG_FCN(object):
    def __init__(self, batch_size, epochs, steps, train_dir, val_dir, learning_rate=0.0001, vgg_path=None, weight_decay=0.):
        self.vgg_path = vgg_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.learning_rate = learning_rate

        content_input = Input(shape=(512, 512, 3))

        # Block 1
        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(content_input)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1_1)
        pool1_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

        # Block 2
        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1_1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2_1)
        pool2_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

        # Block 3
        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2_1)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3_2)
        pool3_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3_3)

        # Block 4
        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3_1)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4_2)
        pool4_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4_3)

        # Block 5
        conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4_1)
        conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5_2)
        pool5_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5_3)

        self.vgg = Model(input=content_input, output=pool5_1)
        self.vgg.load_weights(self.vgg_path, by_name=True)

        fcn_input = self.vgg.output
        # Fully-Connected layers
        conv6 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(fcn_input)
        drop6 = Dropout(0.5)(conv6)
        conv7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(drop6)
        drop7 = Dropout(0.5)(conv7)

        # Classifier
        conv8 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', kernel_regularizer=l2(weight_decay))(drop7)
        output = BilinearUpSampling2D(size=(32, 32))(conv8)

        self.model = Model(input=content_input, output=output)

        self.optimizer = SGD(lr=self.learning_rate, momentum=0.9)
        self.loss_fn = binary_crossentropy_with_logits
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
    
    def train(self):        
        # es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        chkpt = 'model/VGG_FCN_weights.{epoch:02d}-{loss:.2f}.h5'
        cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


        self.model.fit_generator(batch_gen(self.train_dir, self.batch_size),
                            epochs=self.epochs,
                            steps_per_epoch=self.steps,
                            validation_data=next(batch_gen(self.val_dir, self.batch_size)),
                            verbose=1,
                            callbacks=[cp_cb])


    def summary(self):
        # self.vgg.summary()
        self.model.summary()


