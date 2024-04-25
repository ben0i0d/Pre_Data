import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from numpy.lib.format import open_memmap

# uav graph
    # (10, 8), (8, 6), (9, 7), (7, 5), # arms
    # (15, 13), (13, 11), (16, 14), (14, 12), # legs
    # (11, 5), (12, 6), (11, 12), (5, 6), # torso
    # (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears

paris = {
    'v1': ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), 
            (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)) ,

    'v2': ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), 
            (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)) 
}

sets = {
    'train', 'test'
}

datasets = {
    'v1', 'v2',
}

# bone
def gen_bone(dataset, set):
    print(dataset, set)
    data = open_memmap('./data/{}/{}_data_joint.npy'.format(dataset, set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}/{}_data_bone.npy'.format(dataset, set),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    fp_sp[:, :C, :, :, :] = data
    for v1, v2 in tqdm(paris[dataset]):
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

processes = []

for dataset in datasets:
    for set in sets:
        process = multiprocessing.Process(target=gen_bone, args=(dataset, set))
        processes.append(process)
        process.start()

for process in processes:
    process.join()
