from collections import defaultdict
import time
import pickle
import numpy as np
import h5py
from utils.reader import readShortVideo, getVideoList

downsample_factor = 12  # downsample_factor = 12 equals 2fps 

def get_npz(fileType):
    print("Processing {}...\n".format(fileType))
    csv_path = '/home/huaijing/DLCV2018SPRING/hw5/data/TrimmedVideos/label/gt_{}.csv'.format(fileType)
    video_root_path = '/home/huaijing/DLCV2018SPRING/hw5/data/TrimmedVideos/video/{}'.format(fileType)
        
    videoList = getVideoList(csv_path)
    video_name = videoList['Video_name']
    video_cat = videoList['Video_category']
    video_label = videoList['Action_labels']
    video_index = videoList['Video_index']

    length = len(video_label)
    data_frames = defaultdict(np.array)
    data_label = defaultdict(int)

    count = 1

    for i, (index, name, cat, label) in enumerate(zip(video_index, video_name, video_cat, video_label)):
        print('Part {}: {}/{}'.format(count, i+1, length))        
        frames = readShortVideo(video_root_path, cat, name)
        data_frames[str(index)] = frames
        data_label[str(index)] = label

        if fileType == 'train':
            if (int(index)+1) % 809 == 0:  
                data = {'frames':data_frames, 'labels':data_label}
                np.savez_compressed('data/TESTtrain{}.npz'.format(count), **data)
                count += 1

                data_frames = defaultdict(np.array)
                data_label = defaultdict(int)
    if fileType == 'valid':
        data = {'frames':data_frames, 'labels':data_label}
        np.savez_compressed('data/TESTvalid.npz', **data)



def save_npz(frames, labels, dir):
    np.savez_compressed('data/train{}.npz'.format, frames)
    print('Saved as pickle to {}'.format(dir))

def get_h5py(fileType):
    print("Processing {}...\n".format(fileType))
    csv_path = '/home/huaijing/DLCV2018SPRING/hw5/data/TrimmedVideos/label/gt_{}.csv'.format(fileType)
    video_root_path = '/home/huaijing/DLCV2018SPRING/hw5/data/TrimmedVideos/video/{}'.format(fileType)

    limit = 29751 if fileType == 'train' else 4843 
    frames = np.empty((limit, 240, 320, 3), dtype=np.uint8)
    labels = np.empty((limit,), dtype=np.uint8)
    indexes = np.empty((limit,), dtype=np.uint16)

    videoList = getVideoList(csv_path)
    video_name = videoList['Video_name']
    video_cat = videoList['Video_category']
    video_label = videoList['Action_labels']
    video_index = videoList['Video_index']


    count = 0
    for (index, name, cat, label) in zip(video_index, video_name, video_cat, video_label):        
        print("{}/{}".format(count, limit))
        part = readShortVideo(video_root_path, cat, name)
        part_length = part.shape[0]
        start, end = count, count+part_length
        frames[start:end] = part
        labels[start:end] = label
        indexes[start:end] = index

        count += part_length

    with h5py.File('{}.h5'.format(fileType), 'r') as hf:
        start = time.time()
        hf.create_dataset("frames",  data=frames)
        print("write frames in {}".format(time.time()-start))
        hf.create_dataset("labels",  data=labels)
        print("write labels")
        hf.create_dataset("indexes",  data=indexes)
        print("write indexes")


def main():
    # get_h5py('valid')
    get_h5py('train')



if __name__ == '__main__':
    main()


