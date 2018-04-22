from __future__ import print_function, division

import glob
import skimage.color
import skimage.io
import scipy.misc
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.backend as K

import matplotlib.pyplot as plt

color2index = {
    (0  , 255, 255) : 0,
    (255, 255,   0) : 1,
    (255,   0, 255) : 2,
    (0  , 255,   0) : 3,
    (  0,   0, 255) : 4,
    (255, 255, 255) : 5,
    (  0,   0,   0) : 6
}

index2color = {
    0 : (0  , 255, 255),
    1 : (255, 255,   0),
    2 : (255,   0, 255),
    3 : (0  , 255,   0),
    4 : (  0,   0, 255),
    5 : (255, 255, 255),
    6 : (  0,   0,   0)
}

def get_file_paths(dir):
    content = glob.glob("{}/*.jpg".format(dir))
    mask = glob.glob("{}/*.png".format(dir))
    return content, mask

def get_image(path):
    img = skimage.io.imread(path)
    assert len(img.shape) == 3, "# of channels of {} is not 3".format(path)
    return img

def vgg_sub_mean(img):
    img = img.astype(np.float32)
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    return img


def mask_preprocess(img):
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    result[:,:] = -1
    for rgb, idx in color2index.items():
        result[(img==rgb).all(2)] = idx
    one_hot_labels = np.eye(7, dtype=np.float32)[result]
    
    return one_hot_labels

def mask_postprocess(one_hot):
    predict = np.argmax(one_hot, axis=-1)
    height, width = predict.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    for h in range(height):
        for w in range(width):
            mask[h, w] = index2color[predict[h, w]]
    return mask