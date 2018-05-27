import numpy as np
import h5py
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

cnn_path = '/home/huaijing/DLCV2018SPRING/hw5/data/valid_features.h5'
rnn_path = '/home/huaijing/DLCV2018SPRING/hw5/data/rnn_feature_valid.h5'

def plot_tsne(cnn, rnn, labels, state=0):
    print("performing tsne...")
    tsne = TSNE(n_components=2, random_state=state)
    cnn_2d = tsne.fit_transform(cnn)
    rnn_2d = tsne.fit_transform(rnn)
    target_ids = range(11)

    print('Scattering Points....')
    colors = 'g', 'gray', 'b', 'c', 'm', 'y', 'k', 'tan', 'orange', 'purple', 'r'

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(1, 2, 1)
    action = ['Other', 'Inspect/Read', 'Open', 'Take', 'Cut', 'Put', 'Close', 'MoveAround', 'Divide/PullApart', 'Pour', 'Transfer']
    for i, c, label in zip(target_ids, colors, action):
        ax.scatter(cnn_2d[labels == i, 0], cnn_2d[labels == i, 1], c=c, label=label, s=10)
    ax.legend()
    ax.set_title('CNN-based Validation features')

    ax = fig.add_subplot(1, 2, 2)
    for i, c, label in zip(target_ids, colors, action):
        ax.scatter(rnn_2d[labels == i, 0], rnn_2d[labels == i, 1], c=c, label=label, s=10)
    ax.legend()
    ax.set_title('RNN-based Validation features')
    ax.set_xlim(-30, 60)

    plt.show()

with h5py.File(cnn_path, 'r') as hf:
    cnn_frame_features = hf['features'][:]
    labels = hf['labels'][:]
    start = hf['start_idx'][:]
    end = hf['end_idx'][:]    

with h5py.File(rnn_path, 'r') as hf:
    rnn_features = hf['features'][:]

Npoint = labels.shape[0]
cnn_features = np.empty((Npoint, 2048), dtype=np.float32)
for i, (s, e) in enumerate(zip(start, end)):
    cnn_features[i] = np.mean(cnn_frame_features[s:e], axis=0)

for i in range(10):
    plot_tsne(cnn_features, rnn_features, labels, state=i)





