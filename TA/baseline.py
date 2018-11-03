import os
import time
import glob
import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import rescale
from keras.models import Model, load_model
from keras.utils import to_categorical

def load_data(dataType):
    txt_path = "/home/huaijing/DLCV2018SPRING/TA/data/{}_id.txt".format(dataType)
    img_root_path = "/home/huaijing/DLCV2018SPRING/TA/data/{}/".format(dataType)
    with open(txt_path, 'r') as f:
        raw = [line.rstrip('\n').split(' ') for line in f]

    print('Loading {} dataset...'.format(dataType))
    start = time.time()
    filename = [os.path.join(img_root_path, data[0]) for data in raw]
    # images = [rescale(imread(f), scale=1.15) for f in filename]
    images = [imread(f) for f in filename]
    gt_labels = [id2label_dict[int(data[1])] for data in raw]
    images = np.array(images, dtype=np.float32) / 255.0
    print('Load {} in {} secs'.format(dataType, time.time()-start))

    return raw, images, gt_labels

def predict_data(raw, images, gt_labels, filename=None):
    pred_labels = model.predict(images, batch_size=64, verbose=0)
    pred_labels = np.argmax(pred_labels, axis=-1)
    Nsample = pred_labels.shape[0]

    count = 0    
    for i, (gt, pred) in enumerate(zip(gt_labels, pred_labels)):
        # print('raw_id: {} gt_id: {}, pred_id: {}'.format(raw[i][0], raw[i][1], label2id_dict[pred]))
        # print("{}/{} gt: {}, pred: {}".format(i+1, Nsample, gt, pred))
        if filename is not None:
            # write output of Validation only
            with open(os.path.join('result', 'resnet34', filename), 'a+') as f:
                f.write(str(i+1) + ',' + str(label2id_dict[pred]) + '\n')
        if gt == pred:
            count += 1
    return count / Nsample


# n_classes = 2360
# Nval, Ntest = 7211, 7152

id2label_path = "/home/huaijing/DLCV2018SPRING/TA/data/id2label.npy"
label2id_path = "/home/huaijing/DLCV2018SPRING/TA/data/label2id.npy"
id2label_dict = np.load(id2label_path).item()
label2id_dict = np.load(label2id_path).item()

val_raw, val_images, val_gt_labels = load_data('val')
test_raw, test_images, test_gt_labels = load_data('test')

model_root_path = "/home/huaijing/DLCV2018SPRING/TA/ckpt/resnet34/"
model_paths = [path for path in sorted(glob.glob(model_root_path + '*.h5'))]

for i, model_path in enumerate(model_paths, start=1):
    model = load_model(model_path)
    print('\n{:02d}/{} Predicting model {}...'.format(i, len(model_paths), 'finetune_' + model_path.split('.')[1]))

    output_name = 'resnet34_' + model_path.split('.')[1] + '.csv'
    val_acc = predict_data(val_raw, val_images, val_gt_labels)
    test_acc = predict_data(test_raw, test_images, test_gt_labels, output_name)

    print("{} Acc: {:.3f}%".format('Validation', val_acc*100))
    print("{} Acc: {:.3f}%".format('Testing', test_acc*100))