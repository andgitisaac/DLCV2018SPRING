import os
import pickle
import numpy as np

def load_pickle(dir, split='train'):
    pickle_file = 'train.pkl' if split == 'train' else 'valid.pkl'
    with open(os.path.join(dir, pickle_file), 'rb') as f:
        data = pickle.load(f)
    frames = data['video']
    labels = data['label']
    nouns = data['noun']
    print('Finished loading pickle')
    return frames, labels, nouns