from collections import defaultdict
import csv
import pickle
import numpy as np
import pandas as pd
from utils import get_image

def get_pickle(type_):
    print("Processing {}...".format(type_))
    csv_path = 'data/{}.csv'.format(type_)
    data_path = 'data/{}/'.format(type_)
    pkl = defaultdict(list)

    df = pd.read_csv(csv_path, delimiter=',', header=0)
    attrs = df.columns.tolist()

    for idx, path in enumerate(df[attrs[0]].tolist()):
        print("\r{}/{}".format(idx, len(df[attrs[0]].tolist())), end='')
        pkl['images'].append(get_image(data_path + path))
        attr_value = [float((df[x].tolist()[idx])) for x in attrs[1:]]
        attr_dict = dict(zip(attrs[1:], attr_value))
        pkl['attrs'].append(attr_dict)

    pkl['images'] = np.asarray(pkl['images'])
    pkl['attrs'] = np.asarray(pkl['attrs'])
    return pkl

def save_pickle(obj, dir):
    with open(dir, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    print('Saved as pickle to {}'.format(dir))

def main():
    train = get_pickle('train')
    save_pickle(train, 'data/train.pkl')
    
    test = get_pickle('test')
    save_pickle(test, 'data/test.pkl')


    
if __name__ == '__main__':
    main()