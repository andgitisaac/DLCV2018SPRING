import time
import numpy as np
import h5py
from keras.models import Model, load_model
from utils.reader import readShortVideo, getVideoList

downsample_factor = 12  # downsample_factor = 12 equals 2fps 


def get_h5py(fileType, model):
    print("Processing {}...\n".format(fileType))
    csv_path = '/home/huaijing/DLCV2018SPRING/hw5/data/TrimmedVideos/label/gt_{}.csv'.format(fileType)
    video_root_path = '/home/huaijing/DLCV2018SPRING/hw5/data/TrimmedVideos/video/{}'.format(fileType)

    totalFrame = 29751 if fileType == 'train' else 4843
    totalVideo = 3236 if fileType == 'train' else 517
    features = np.empty((totalFrame, 2048), dtype=np.float32)
    labels = np.empty((totalVideo,), dtype=np.uint8)
    start_idx = np.empty((totalVideo,), dtype=np.uint16)
    end_idx = np.empty((totalVideo,), dtype=np.uint16)

    videoList = getVideoList(csv_path)
    video_name = videoList['Video_name']
    video_cat = videoList['Video_category']
    video_label = videoList['Action_labels']

    count = 0
    with h5py.File('data/{}_features_norm.h5'.format(fileType), 'w') as hf:
        for i, (name, cat, label) in enumerate(zip(video_name, video_cat, video_label)):        
            print("{}/{} {}/{}".format(i, totalVideo, count, totalFrame))
            part = readShortVideo(video_root_path, cat, name)
            part = (part / 127.5) - 1.0
            part_length = part.shape[0]
            start, end = count, count+part_length
            
            features[start:end] = model.predict(part, batch_size=4)
            labels[i] = label
            start_idx[i] = start
            end_idx[i] = end

            count += part_length

        
        start = time.time()
        hf.create_dataset("features",  data=features)
        print("write features in {}".format(time.time()-start))
        hf.create_dataset("labels",  data=labels)
        print("write labels")
        hf.create_dataset("start_idx",  data=start_idx)
        print("write start_idx")
        hf.create_dataset("end_idx",  data=end_idx)
        print("write end_idx")


def main():
    resnet = load_model('ckpt/resnetBest_frame.h5')

    resnet_input = resnet.get_layer(name='input_1').input
    feature_output = resnet.get_layer(name='global_average_pooling2d_1').output
    base_model = Model(input=resnet_input, output=feature_output)
    get_h5py('valid', base_model)
    get_h5py('train', base_model)



if __name__ == '__main__':
    main()


