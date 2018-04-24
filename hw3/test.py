from __future__ import print_function, division

import glob
import argparse
import skimage.io
import numpy as np
from utils import get_image, mask_postprocess, vgg_sub_mean
from ops import batch_gen
from model import VGG_FCN32, VGG_FCN8

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str,
                    dest='net', help='kinds of network',
                    required=True)
parser.add_argument('--input-path', type=str,
                    dest='input_path', help='testing images directory',
                    required=True)
parser.add_argument('--output-path', type=str,
                    dest='output_path', help='output images directory',
                    required=True)
parser.add_argument('--model-path', type=str,
                    dest='model_path', help='directory of weights of the entire model',
                    required=True)
# parser.add_argument('--ep-h5', type=int,
#                     dest='ep', help='Epoch number of h5 files',
#                     required=True)
args = parser.parse_args()

if __name__ == '__main__':
    test_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    batch_size = 4

    if args.net == 'fcn32':
        model = VGG_FCN32(mode='test', model_path=model_path)
    elif args.net == 'fcn8':
        model = VGG_FCN8(mode='test', model_path=model_path)
    

    content_path = glob.glob("{}/*.jpg".format(test_path))
    content_path.sort()

    for i, path in enumerate(content_path):
        output_name = "{}{:04d}_mask.png".format(output_path, i)

        img = vgg_sub_mean(get_image(path))
        img = np.expand_dims(img, axis=0)
        reconstructed_mask = model.decode(img)
        reconstructed_mask = mask_postprocess(reconstructed_mask[0])
        skimage.io.imsave(output_name, reconstructed_mask)
    






