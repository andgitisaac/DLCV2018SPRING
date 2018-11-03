import os
import time
import skimage
import skimage.io
import numpy as np
import h5py

def read_txt(path):
    with open(path, 'r') as f:
        line = [line.rstrip('\n').split(' ') for line in f]
        return line


def get_h5py(fileType):
    print("Processing {}...\n".format(fileType))
    txt_path = "/home/huaijing/DLCV2018SPRING/TA/data/{}_id.txt".format(fileType)
    img_root_path = "/home/huaijing/DLCV2018SPRING/TA/data/{}/".format(fileType)

    raw = read_txt(txt_path)
    Nsample = len(raw)

    # images = np.empty((Nsample, 218, 178, 3), dtype=np.uint8)
    # labels = np.empty((Nsample,), dtype=np.uint16)

    
    labels = [int(sample[1]) for sample in raw]
    unique_labels = list(set(labels))
    seq = list(range(Nsample))
    id2label_dict = dict(zip(unique_labels, seq))
    label2id_dict = dict(zip(seq, unique_labels))
    np.save('id2label.npy', id2label_dict) 
    np.save('label2id.npy', label2id_dict) 



    # with h5py.File('data/{}.h5'.format(fileType), 'w') as hf:
    #     for i, sample in enumerate(raw):
    #         print('{}\{}'.format(i+1, Nsample))
    #         filename, label = sample[0], sample[1]
    #         filename = os.path.join(img_root_path, filename)
    #         images[i] = skimage.io.imread(filename)
    #         labels[i] = label
        
    #     hf.create_dataset("images",  data=images)
    #     hf.create_dataset("labels",  data=labels)


def main():    
    # get_h5py('test')
    # get_h5py('val')
    get_h5py('train')



if __name__ == '__main__':
    main()


