import os
import pickle
import numpy as np
import skimage.io

def get_image(path):
    img = skimage.io.imread(path)
    assert len(img.shape) == 3, "# of channels of {} is not 3".format(path)
    return img

def load_pickle(dir, split='train'):
    img_file = 'train.pkl' if split == 'train' else 'test.pkl'
    with open(os.path.join(dir, img_file), 'rb') as f:
        celebA = pickle.load(f)
    images = celebA['images'] / 127.5 - 1
    attrs = celebA['attrs']
    print('Finished loading celebA')
    return images, attrs