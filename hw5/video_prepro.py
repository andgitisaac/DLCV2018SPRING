from collections import defaultdict
import pickle
import numpy as np
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

    length = len(video_label)
    data_frames = defaultdict(np.array)
    data_label = defaultdict(int)

    count = 1

    for index, (name, cat, label) in enumerate(zip(video_name, video_cat, video_label)):
        print('Part {}: {}/{}'.format(count, index+1, length))        
        frames = readShortVideo(video_root_path, cat, name)
        data_frames[str(index)] = frames
        data_label[str(index)] = label

        if (index+1) % 809 == 0:  
            data = {'frames':data_frames, 'labels':data_label}
            np.savez_compressed('data/train{}.npz'.format(count), **data)
            count += 1

            data_frames = defaultdict(np.array)
            data_label = defaultdict(int)


def save_npz(frames, labels, dir):
    np.savez_compressed('data/train{}.npz'.format, frames)
    print('Saved as pickle to {}'.format(dir))

def main():
    # data = get_npz('valid')
    # np.savez_compressed('data/valid.npz', **data)
    get_npz('train')
    # np.savez_compressed('data/train1.npz', **data1)
    # np.savez_compressed('data/train2.npz', **data2)



if __name__ == '__main__':
    main()


