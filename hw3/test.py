from __future__ import print_function, division

import glob
import argparse
import skimage.io
import numpy as np
from utils import get_image, mask_postprocess, vgg_sub_mean
from ops import batch_gen
from model import VGG_FCN32, VGG_FCN8, VGG_UNET

parser = argparse.ArgumentParser()
parser.add_argument('--ep-h5', type=int,
                    dest='ep', help='Epoch number of h5 files',
                    required=True)
args = parser.parse_args()

if __name__ == '__main__':
    test_path = 'data/validation/'
    output_path = 'output/'
    # model_path = 'model/best/VGG_FCN8.{:02d}.h5'.format(args.ep)
    model_path = 'model/VGG_UNET.{:02d}.h5'.format(args.ep)
    vgg_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    batch_size = 4

    # vgg = VGG_FCN32(batch_size, mode='test', model_path=model_path, vgg_path=vgg_path)  
    # vgg =  VGG_FCN8(batch_size, mode='test', model_path=model_path, vgg_path=vgg_path)    
    vgg = VGG_UNET(batch_size, mode='test', model_path=model_path, vgg_path=vgg_path)

    vgg.plot()

    print("----------------{}-----------------".format(model_path))
    content_path = glob.glob("{}/*.jpg".format(test_path))
    content_path.sort()

    for i, path in enumerate(content_path):
        output_name = "{}{:04d}_mask.png".format(output_path, i)
        # print(path, output_name)

        img = vgg_sub_mean(get_image(path))
        img = np.expand_dims(img, axis=0)
        reconstructed_mask = vgg.decode(img)
        reconstructed_mask = mask_postprocess(reconstructed_mask[0])
        skimage.io.imsave(output_name, reconstructed_mask)
    






