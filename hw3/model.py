from __future__ import print_function, division

import numpy as np
from ops import batch_gen
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation, Conv2DTranspose, Concatenate, BatchNormalization, Cropping2D, Add
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.models import load_model

class VGG_FCN32(object):
    def __init__(self,
                batch_size,
                mode='train',
                epochs=None,
                steps=None,
                learning_rate=0.0001,
                train_dir=None,
                val_dir=None,            
                vgg_path=None,
                model_path=None,                
                weight_decay=0.):

        self.mode = mode
        self.vgg_path = vgg_path
        self.model_path = model_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.learning_rate = learning_rate


        ### Build VGG-16 ###
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

        # self.vgg = Model(input=content_input, output=pool5_1)

        # if self.mode == 'train':
        #     self.vgg.load_weights(self.vgg_path, by_name=True)
        #     for layer in self.vgg.layers:
        #         layer.trainable = False


        ### Build FCN-32s ###
        fcn_input = pool5_1
        # fcn_input = self.vgg.output

        # Fully-Connected layers
        conv6 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fcn_fc1')(fcn_input)
        drop6 = Dropout(0.5)(conv6)
        conv7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fcn_fc2')(drop6)
        drop7 = Dropout(0.5)(conv7)

        # Classifier
        conv8 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', activation='linear', padding='valid')(drop7)
        up8 = Conv2DTranspose(7, (64, 64), strides=(32, 32), padding='same', name='deconv')(conv8)
        output = Activation('softmax')(up8)

        self.model = Model(input=content_input, output=output)

        if self.mode == 'train':
            self.model.load_weights(self.vgg_path, by_name=True)
            # for i, layer in enumerate(self.model.layers):
            #     if i < 19:
            #         layer.trainable = False


            self.summary()
            self.optimizer = Adam(1e-4)
            # self.optimizer = 'sgd'
            self.model.compile(optimizer=self.optimizer,
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
        elif self.mode == 'test':
            self.load(self.model_path)
            print("------------Weight Loaded--------------")
    
    def train(self):        
        # es_cb = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
        chkpt = 'model/VGG_FCN32_weights.{epoch:02d}.h5'
        # cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        cp_cb = ModelCheckpoint(filepath = chkpt, monitor='loss', verbose=1, save_best_only=False, mode='auto')


        self.model.fit_generator(batch_gen(self.train_dir, self.batch_size),
                            epochs=self.epochs,
                            steps_per_epoch=self.steps,
                            # validation_data=next(batch_gen(self.val_dir, self.batch_size)),
                            verbose=1,
                            callbacks=[cp_cb])

    def plot(self):
        plot_model(self.model, to_file='VGG16_FCN32s.png')

    def summary(self):
        # self.vgg.summary()
        self.model.summary()

    def load(self, path):
        self.model.load_weights(path, by_name=True)
        # self.model.summary()
        
    
    def decode(self, x):
        return self.model.predict(x)


class VGG_FCN8(object):
    def __init__(self,
                batch_size,
                mode='train',
                epochs=None,
                steps=None,
                learning_rate=0.0001,
                train_dir=None,
                val_dir=None,            
                vgg_path=None,
                model_path=None,                
                weight_decay=0.):

        self.mode = mode
        self.vgg_path = vgg_path
        self.model_path = model_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.learning_rate = learning_rate


        ### Build VGG-16 ###
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

        if self.mode == 'train':
            self.vgg.load_weights(self.vgg_path, by_name=True)
            # for layer in self.vgg.layers:
            #     layer.trainable = False


        ### Build FCN-8s ###
        fcn_input = pool5_1
        # fcn_input = self.vgg.output

        # Fully-Connected layers
        conv6 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fcn_fc1')(fcn_input)
        drop6 = Dropout(0.5)(conv6)
        conv7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fcn_fc2')(drop6)
        drop7 = Dropout(0.5)(conv7)

        # Classifier
        conv8 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', padding='valid', name='conv8')(drop7)        
        up8 = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='same', use_bias=False, name='deconv8_1')(conv8)

        scaled_pool4 = Lambda(lambda x: x * 0.01, name='scaled_pool4')(pool4_1)      
        conv9 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', padding='valid', name='conv9')(scaled_pool4)
        o1, o2 = self.crop(up8, conv9, content_input)
        add9 = Add()([o1, o2])
        up9 = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='same', use_bias=False, name='deconv9_1')(add9)

        scaled_pool3 = Lambda(lambda x: x * 0.0001, name='scaled_pool3')(pool3_1)  
        conv10 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', padding='valid', name='conv10')(scaled_pool3)
        o1, o2 = self.crop(up9, conv10, content_input)
        add10 = Add()([o1, o2])

        up11 = Conv2DTranspose(7, (16, 16), strides=(8, 8), padding='same', use_bias=False, name='deconv10_1')(add10)
        output = Activation('softmax')(up11)

        self.model = Model(input=content_input, output=output)

        if self.mode == 'train':
            self.model.load_weights(self.vgg_path, by_name=True)
            for i, layer in enumerate(self.model.layers):
                if i < 19:
                    layer.trainable = False


            self.summary()
            # self.optimizer = Adam(1e-4)
            self.optimizer = 'adadelta'
            # self.optimizer = 'sgd'
            self.model.compile(optimizer=self.optimizer,
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
        elif self.mode == 'test':
            self.load(self.model_path)
            print("------------Weight Loaded--------------")
    
    def crop(self, o1, o2, input):
        o1_shape = Model(input=input, output=o1).output_shape
        o1_height = o1_shape[1]
        o1_width = o1_shape[2]

        o2_shape = Model(input=input, output=o2).output_shape
        o2_height = o2_shape[1]
        o2_width = o2_shape[2]

        cx = abs(o1_width - o2_width)
        cy = abs(o1_height - o2_height)

        if (o1_width > o2_width):
            o1 = Cropping2D(cropping=((0, 0) ,  (0, cx)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, 0) ,  (0, cx)))(o2)
        if (o1_height > o2_height):
            o1 = Cropping2D(cropping=((0, cy) ,  (0, 0)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, cy) ,  (0, 0)))(o2)
        
        return o1, o2


    
    def train(self):        
        # es_cb = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
        chkpt = 'model/VGG_FCN8_weights.{epoch:02d}.h5'
        # cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        cp_cb = ModelCheckpoint(filepath = chkpt, monitor='loss', verbose=1, save_best_only=False, mode='auto')


        self.model.fit_generator(batch_gen(self.train_dir, self.batch_size),
                            epochs=self.epochs,
                            steps_per_epoch=self.steps,
                            # validation_data=next(batch_gen(self.val_dir, self.batch_size)),
                            verbose=1,
                            callbacks=[cp_cb])

    def plot(self):
        plot_model(self.model, to_file='VGG16_FCN8s.png')

    def summary(self):
        # self.vgg.summary()
        self.model.summary()

    def load(self, path):
        self.model.load_weights(path, by_name=True)
        # self.model.summary()
        
    
    def decode(self, x):
        return self.model.predict(x)



class VGG_UNET(object):
    def __init__(self,
                batch_size,
                mode='train',
                epochs=None,
                steps=None,
                learning_rate=0.0001,
                train_dir=None,
                val_dir=None,            
                vgg_path=None,
                model_path=None,                
                weight_decay=0.):

        self.mode = mode
        self.vgg_path = vgg_path
        self.model_path = model_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.learning_rate = learning_rate
        
        ### Build VGG-16 ###
        content_input = Input(shape=(512, 512, 3))

        # # Block 1
        # x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(content_input)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)        
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # # Block 2
        # x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(content_input)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)        
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # # Block 3
        # x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(content_input)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)      
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # # Block 4
        # x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(content_input)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # # Block 5
        # x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(content_input)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
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

        if self.mode == 'train':
            self.vgg.load_weights(self.vgg_path, by_name=True)
            for layer in self.vgg.layers:
                layer.trainable = False

        ### Build Decoder ###
        x = pool5_1
        x = Dropout(0.5)(x)

        x = UpSampling2D(size=(2, 2), name='block6_up')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='block6_conv1')(x)
        x = BatchNormalization(name='block6_BN1')(x)
        x = Activation('relu', name='block6_relu1')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='block6_conv2')(x)
        x = BatchNormalization(name='block6_BN2')(x)
        x = Activation('relu', name='block6_relu2')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='block6_conv3')(x)
        x = BatchNormalization(name='block6_BN3')(x)
        x = Activation('relu', name='block6_relu3')(x)
        x = Dropout(0.5)(x)

        x = UpSampling2D(size=(2, 2), name='block7_up')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv1')(x)
        x = BatchNormalization(name='block7_BN1')(x)
        x = Activation('relu', name='block7_relu1')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv2')(x)
        x = BatchNormalization(name='block7_BN2')(x)
        x = Activation('relu', name='block7_relu2')(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='block7_conv3')(x)
        x = BatchNormalization(name='block7_BN3')(x)
        x = Activation('relu', name='block7_relu3')(x)
        x = Dropout(0.5)(x)

        x = UpSampling2D(size=(2, 2), name='block8_up')(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv1')(x)
        x = BatchNormalization(name='block8_BN1')(x)
        x = Activation('relu', name='block8_relu1')(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv2')(x)
        x = BatchNormalization(name='block8_BN2')(x)
        x = Activation('relu', name='block8_relu2')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block8_conv3')(x)
        x = BatchNormalization(name='block8_BN3')(x)
        x = Activation('relu', name='block8_relu3')(x)
        x = Dropout(0.5)(x)

        x = UpSampling2D(size=(2, 2), name='block9_up')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='block9_conv1')(x)
        x = BatchNormalization(name='block9_BN1')(x)
        x = Activation('relu', name='block9_relu1')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block9_conv2')(x)
        x = BatchNormalization(name='block9_BN2')(x)
        x = Activation('relu', name='block9_relu2')(x)

        x = UpSampling2D(size=(2, 2), name='block10_up')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block10_conv1')(x)
        x = BatchNormalization(name='block10_BN1')(x)
        x = Activation('relu', name='block10_relu1')(x)
        # x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='block10_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        x = Conv2D(7, (1, 1), padding = 'same', kernel_initializer = 'he_normal', name='output')(x)
        output = Activation('softmax')(x)

        self.model = Model(input=content_input, output=output)

        if self.mode == 'train':
            self.summary()
            # self.optimizer = 'adadelta'
            self.optimizer = 'adam'
            # self.optimizer = SGD(lr=0.0001, momentum=0.9)
            self.model.compile(optimizer=self.optimizer,
                                # loss='mse',
                                loss='categorical_crossentropy',
                                # loss=self.loss_fn,
                                metrics=['accuracy'])
        elif self.mode == 'test':
            self.load(self.model_path)
            print("------------Weight Loaded--------------")
    
    def train(self):        
        # es_cb = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
        chkpt = 'model/VGG_UNET.{epoch:02d}.h5'
        # cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        cp_cb = ModelCheckpoint(filepath = chkpt, monitor='loss', verbose=1, save_best_only=False, mode='auto')


        self.model.fit_generator(batch_gen(self.train_dir, self.batch_size),
                            epochs=self.epochs,
                            steps_per_epoch=self.steps,
                            # validation_data=next(batch_gen(self.val_dir, self.batch_size)),
                            verbose=1,
                            callbacks=[cp_cb])


    def summary(self):
        # self.vgg.summary()
        self.model.summary()

    def load(self, path):
        self.model.load_weights(path, by_name=True)
        # self.model.summary()

    def plot(self):
        plot_model(self.model, to_file='VGG16_UNET.png') 
    
    def decode(self, x):
        return self.model.predict(x)
    


