import os
import pickle
import numpy as np
import skimage.io
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

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

def postpro(image_arr):
    image_arr = (image_arr + 1) * 127.5
    image_arr = image_arr.astype('uint8')
    return image_arr

def grid_plot_pair(original, reconst, n_col, output_name):
    batch_size, height, width, _ = original.shape
    assert batch_size <= 32, 'batch size should not exceed 32'
    assert batch_size % n_col == 0, 'Adjust n_col {} to be a factor of batch size {}'.format(n_col, batch_size)
    
    n_row = batch_size // n_col
    grid = Image.new('RGB', (2*n_row*height, n_col*width))

    original = postpro(original)
    reconst = postpro(reconst)

    for i in range(batch_size):
        ori_offset_h = 2 * (i // n_col) * height
        rec_offset_h = (2 * (i // n_col) + 1) * height
        offset_w = i % n_col * width

        ori = Image.fromarray(original[i])   
        rec = Image.fromarray(reconst[i])

        grid.paste(ori, (offset_w, ori_offset_h, offset_w+width, ori_offset_h+height))
        grid.paste(rec, (offset_w, rec_offset_h, offset_w+width, rec_offset_h+height))
    grid.save(output_name)

def grid_plot(image, n_col, output_name):
    batch_size, height, width, _ = image.shape
    assert batch_size <= 32, 'batch size should not exceed 32'
    assert batch_size % n_col == 0, 'Adjust n_col {} to be a factor of batch size {}'.format(n_col, batch_size)
    
    n_row = batch_size // n_col
    grid = Image.new('RGB', (n_col*width, n_row*height))

    image = postpro(image)

    for i in range(batch_size):
        offset_h = (i // n_col) * height
        offset_w = (i % n_col) * width

        img = Image.fromarray(image[i])   

        grid.paste(img, (offset_w, offset_h, offset_w+width, offset_h+height))
    grid.save(output_name)

def plot_tsne(encode, attrs):
    print("performing tsne...")
    tsne = TSNE(n_components=2, random_state=0)
    encode_2d = tsne.fit_transform(encode)

    target_attr = 'Male'
    is_target = []
    not_target = []
    for i in range(attrs.shape[0]):
        if attrs[i][target_attr] == 1:
            is_target.append(i)
        else:
            not_target.append(i)

    fig = plt.figure(figsize=(8, 6))   
    plt.scatter(encode_2d[is_target, 0], encode_2d[is_target, 1], s=6, c='r', label='Male')
    plt.scatter(encode_2d[not_target, 0], encode_2d[not_target, 1], s=6, c='b', label='Female')
        
    plt.legend()
    fig.savefig('vae_encode.png')
    plt.close()


