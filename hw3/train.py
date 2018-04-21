from __future__ import print_function, division

import numpy as np
from ops import batch_gen
from model import VGG_FCN

if __name__ == '__main__':
    train_path = 'data/train/'
    val_path = 'data/validation'
    vgg_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    batch_size = 3
    steps = 2313 // batch_size
    epochs = 10

    vgg_fcn = VGG_FCN(batch_size, epochs, steps, train_path, val_path, vgg_path=vgg_path)
    vgg_fcn.summary()

    vgg_fcn.train()


    # while True:
    #     content, mask = next(batch_gen(train_path, batch_size))
    #     print(content.shape, mask.shape)