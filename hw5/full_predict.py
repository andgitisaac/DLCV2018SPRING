import os
import pandas as pd
import skimage
from skimage.io import imread
import numpy as np
from keras.models import Model, load_model

cnn_path = "/home/huaijing/DLCV2018SPRING/hw5/model/p3/resnet.h5"
rnn_path = "/home/huaijing/DLCV2018SPRING/hw5/model/p3/full.h5"
root_dir = "/home/huaijing/DLCV2018SPRING/hw5/data/FullLengthVideos/"
video_root_dir = os.path.join(root_dir, 'videos')
label_root_dir = os.path.join(root_dir, 'labels')

def list_files(root):
    r = []    
    dirs = os.listdir(root)
    for dir in sorted(dirs):
        for _, _, files in os.walk(os.path.join(root, dir)):
            lst = []
            for f in sorted(files):
                lst.append(os.path.join(root, dir, f))
            r.append(lst)
    return r

def read_labels(root):
    files = sorted(os.listdir(root))
    lab = []
    for file in files:
        file_path = os.path.join(root, file)
        data = pd.read_csv(file_path, header=None)        
        lab.append(data.ix[:, 0].tolist())
    return lab

base_model = load_model(cnn_path)
cnn_input = base_model.get_layer(name='input_1').input
feature_output = base_model.get_layer(name='global_average_pooling2d_1').output
cnn = Model(input=cnn_input, output=feature_output)

rnn = load_model(rnn_path)
# rnn.summary()

max_length = 400
valid_label_list = read_labels(os.path.join(label_root_dir, 'valid'))
valid_file_list = list_files(os.path.join(video_root_dir, 'valid'))




total_count, total_video_length = 0, 0
for video_count, (video_sublist, label_sublist) in enumerate(zip(valid_file_list, valid_label_list)):
    single_count = 0
    video_length = len(video_sublist)
    print("{}/{} Video Length: {}".format(video_count+1, 5, video_length))

    
    loopTime = video_length // max_length        
    whole_pred_label = np.zeros(((loopTime+1)*max_length,), dtype=np.uint8)

    whole_features = np.zeros(((loopTime+1)*max_length, 2048), dtype=np.float32)
    for steps in range(loopTime+1):
        frames = np.zeros((max_length, 240, 320, 3), dtype=np.uint8)
        labels = np.zeros((max_length,), dtype=np.uint8)
        iter_step = (video_length-steps*max_length) if (steps == loopTime) else max_length

        for i in range(iter_step):
            f = video_sublist[steps*max_length+i]            
            frames[i] = imread(f)
            labels[i] = label_sublist[steps*max_length+i]

        features = cnn.predict(frames, batch_size=16)
        whole_features[steps*max_length:(steps+1)*max_length] = features
    whole_features = whole_features[:video_length]

    pred_label = rnn.predict(np.expand_dims(whole_features, axis=0))
    pred_label = np.argmax(pred_label, axis=-1)

    np.save('model/p3/video_{}.npy'.format(video_count+1), pred_label[0])

    for target, pred in zip(label_sublist, pred_label[0]):
        if target == pred:
            single_count += 1

    total_video_length += video_length
    total_count += single_count
    print("Video {} Acc: {:.5f}".format(video_count+1, single_count/video_length))
print("Total Acc: {:.5f}".format(total_count/total_video_length))