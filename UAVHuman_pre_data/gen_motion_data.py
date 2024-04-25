import os
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

sets = {
    'train', 'test'
}

datasets = {
    'v1', 'v2',
}

parts = {
    'joint', 'bone'
}


def gen_motion(dataset, set,part):
    print(dataset, set, part)
    data = open_memmap('./data/{}/{}_data_{}.npy'.format(dataset, set, part),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}/{}_data_{}_motion.npy'.format(dataset, set, part),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
    fp_sp[:, :, T - 1, :, :] = 0

for dataset in datasets:
    for set in sets:
        for part in parts:
            gen_motion(dataset, set, part)
