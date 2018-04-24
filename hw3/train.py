from __future__ import print_function, division

import argparse
import numpy as np
from ops import batch_gen
from model import VGG_FCN32, VGG_FCN8

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str,
                    dest='net', help='kinds of network',
                    required=True)
args = parser.parse_args()

if __name__ == '__main__':
    train_path = 'data/train/'
    val_path = 'data/validation/'
    vgg_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

    if args.net == 'fcn32':
        batch_size = 3
        steps = 2313 // batch_size
        epochs = 20
        model = VGG_FCN32(batch_size, mode='train', epochs=epochs, steps=steps, train_dir=train_path, val_dir=val_path, vgg_path=vgg_path)
    elif args.net == 'fcn8':
        batch_size = 3
        steps = 2313  // batch_size
        epochs = 40
        model = VGG_FCN8(batch_size, mode='train', epochs=epochs, steps=steps, train_dir=train_path, val_dir=val_path, vgg_path=vgg_path)

    model.train()

