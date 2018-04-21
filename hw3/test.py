from __future__ import print_function, division

import glob
import skimage.io
import numpy as np
from utils import get_image, mask_postprocess, vgg_sub_mean
from ops import batch_gen
from model import VGG_FCN32, VGG_UNET

import matplotlib.pyplot as plt


if __name__ == '__main__':
    test_path = 'data/validation/'
    output_path = 'output/'
    model_path = 'model/VGG_UNET_weights.24-0.59.h5'
    vgg_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    batch_size = 4

    # vgg = VGG_FCN32(batch_size, mode='test', model_path=model_path, vgg_path=vgg_path)  
    vgg =  VGG_FCN8(batch_size, mode='train', epochs=epochs, steps=steps, train_dir=train_path, val_dir=val_path, vgg_path=vgg_path)    
    # vgg = VGG_UNET(batch_size, mode='test', model_path=model_path, vgg_path=vgg_path)    

    content_path = glob.glob("{}/*.jpg".format(test_path))
    content_path.sort()
    for i, path in enumerate(content_path):
        output_name = "{}{:04d}_mask.png".format(output_path, i)
        print(path, output_name)

        img = vgg_sub_mean(get_image(path))
        img = np.expand_dims(img, axis=0)
        reconstructed_mask = vgg.decode(img)
        # reconstructed_mask = vgg.decode(img)
        reconstructed_mask = mask_postprocess(reconstructed_mask[0])
        skimage.io.imsave(output_name, reconstructed_mask)




        # fig = plt.figure(figsize=(8, 4))
        # # display original
        # # ax = plt.subplot(1, 2, 1)
        # # plt.imshow(img[0])
        # # ax.set_title("Original Image")
        # # ax.get_xaxis().set_visible(False)
        # # ax.get_yaxis().set_visible(False)

        # ax = plt.subplot(1, 2, 1)
        # plt.imshow(img[0])
        # ax.set_title("Original Mask {}".format(path))
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        
        # # display reconstruction
        # ax = plt.subplot(1, 2, 2)
        # plt.imshow(reconstructed_mask)
        # ax.set_title("Reconstructed {}".format(output_name))
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.show()



