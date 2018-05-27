import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

label_path = "/home/huaijing/DLCV2018SPRING/hw5/data/FullLengthVideos/labels/valid"
im_root = "data/FullLengthVideos/videos/valid/OP02-R04-ContinentalBreakfast/"

def read_labels(root):
    files = sorted(os.listdir(root))
    lab = []
    for file in files:
        file_path = os.path.join(root, file)
        data = pd.read_csv(file_path, header=None)        
        lab.append(data.ix[:, 0].tolist())
    return lab

def get_lineColor(label, x, height):    
    cmap = ListedColormap(['g', 'gray', 'b', 'c', 'm', 'y', 'k', 'tan', 'orange', 'purple', 'r'])
    norm = BoundaryNorm([x-0.5 for x in range(0, 12)], cmap.N)

    y = [height] * x.shape[0] 
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(label)
    lc.set_linewidth(40)

    return lc


im_list = sorted(os.listdir(im_root))
start, end = 560, 860
im_list = im_list[start:end:50]


im = np.zeros((240, 320*len(im_list), 3), dtype=np.uint8)
for i, path in enumerate(im_list):
    im[:, i*320:(i+1)*320, :] = plt.imread(os.path.join(im_root, path))

gt_label = np.array(read_labels(label_path)[1])
pred_label = np.load('model/p3/video_2.npy')
gt_label = gt_label[start:end]
pred_label = pred_label[start:end]
print(gt_label[::50])
length = gt_label.shape[0]

x = np.arange(length)
gt_lc = get_lineColor(gt_label, x, 0.8)
pred_lc = get_lineColor(pred_label, x, 0.2)

fig = plt.figure(figsize=(18, 6))
plt.gca().add_collection(gt_lc)
plt.gca().add_collection(pred_lc)
plt.xlim(x.min(), x.max())
plt.ylim(0, 1)
plt.axis('off')

newax = fig.add_axes([0.13, 0.3, 0.77, 0.4])
newax.imshow(im)
newax.axis('off')

plt.show()