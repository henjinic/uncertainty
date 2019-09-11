"""common.py

0. common functions and constants

Hyeonjin Kim
2019.08.27
"""
from enum import IntEnum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import time

BASE_DIR = './data'

DISASTER_NAMES = 'fire landslide_re flood_re health product'.split()
FEATURE_NAMES = 'geo_re pop prcp tavg'.split()

N_DISASTER = len(DISASTER_NAMES)
N_FEATURE = len(FEATURE_NAMES)

N_COLS = 643
N_ROWS = 661

N_CLASS = 4
A_CLASS = [2, 16, 18]
B_CLASS = [4, 8, 12]
D_CLASS = [0]

HEADER = '''ncols         643
nrows         661
xllcorner     8515.9999992026
yllcorner     47670.999937602
cellsize      1000.0000000944
NODATA_value  -99
'''

class Category(IntEnum):
    SAFE = 0
    DANGER = 1
    UNCERTAIN = 2
    NODATA = 9

def load_data(path, skip=0, sep=None):
    with open(path) as f:
        data = [list(map(float, row.split(sep))) for row in f.readlines()[skip:]]
    return np.array(data)

def save_data(rows, path):
    with open(path, 'w') as f:
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')

def save_map(rows, path):
    with open(path, 'w') as f:
        f.write(HEADER)
        for row in rows:
            f.write(' '.join(map(str, row)) + '\n')

def categorize(prob_vec, lower, upper, nodata_to_uncertain=False):
    result = []
    for elm in prob_vec:
        if elm <= lower and elm >= 0:
            result.append(Category.SAFE)
        elif elm >= upper and elm <= 1:
            result.append(Category.DANGER)
        elif elm == -99:
            result.append(Category.NODATA if not nodata_to_uncertain else Category.UNCERTAIN)
        else:
            result.append(Category.UNCERTAIN)
    return list(map(int, result))

def is_certain(categorized_vec):
    if Category.NODATA in categorized_vec or Category.UNCERTAIN in categorized_vec:
        return False
    else:
        return True

def classify(binary_vec):
    code = int(''.join(map(str, binary_vec)), 2)
    if code in A_CLASS:
        return 0
    elif code in B_CLASS:
        return 1
    elif code in D_CLASS:
        return 3
    else:
        return 2

def onehot_encode(features, ith):
    target_column = features[:, ith]
    uniques, labels = np.unique(target_column, return_inverse=True)

    result = []
    for label in labels:
        onehot_vec = [0.0] * (len(uniques) - 1)

        if label != (len(uniques) - 1):
            onehot_vec[label] = 1.0
        result.append(onehot_vec)

    return np.concatenate((features[:, :ith], result, features[:, ith+1:]), axis=1)

def split(trainset, seed):
    x = trainset[:, :-1]
    y = trainset[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    return x_train, x_test, y_train, y_test

def normalize(baseset, targetset):
    scaler = MinMaxScaler()
    scaler.fit(baseset)
    normalized_baseset = scaler.transform(baseset)
    normalized_targetset = scaler.transform(targetset)
    return normalized_baseset, normalized_targetset

def get_curreut_time():
    return time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

def numbering(path, num):
    tokens = path.split('.')
    return '.'.join(tokens[:-1]) + f'_{num}.' + tokens[-1]

def plot_hist(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()