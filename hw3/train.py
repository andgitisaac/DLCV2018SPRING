from __future__ import print_function, division

import numpy as np
from ops import batch_gen
from model import VGG_FCN32, VGG_FCN8, VGG_UNET

if __name__ == '__main__':
    train_path = 'data/train/'
    val_path = 'data/validation/'
    vgg_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    batch_size = 3
    steps = 2313 // batch_size
    epochs = 25

    # vgg_fcn = VGG_FCN32(batch_size, mode='train', epochs=epochs, steps=steps, train_dir=train_path, val_dir=val_path, vgg_path=vgg_path)
    # vgg_fcn.train()

    vgg_fcn = VGG_FCN8(batch_size, mode='train', epochs=epochs, steps=steps, train_dir=train_path, val_dir=val_path, vgg_path=vgg_path)
    vgg_fcn.train()

    # vgg_unet = VGG_UNET(batch_size, mode='train', epochs=epochs, steps=steps, train_dir=train_path, val_dir=val_path, vgg_path=vgg_path)
    # vgg_unet.train()

