import numba
import psutil
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

from get_angle import getThridOrderRep

parser = argparse.ArgumentParser(description='Dataset Preprocessing')
parser.add_argument('--modal', type=str, default='bone', help='gen_modal')

# uav graph
    # (10, 8), (8, 6), (9, 7), (7, 5), # arms
    # (15, 13), (13, 11), (16, 14), (14, 12), # legs
    # (11, 5), (12, 6), (11, 12), (5, 6), # torso
    # (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears

datasets = {'v1', 'v2'}
sets = {'train', 'test'}
parts = {'joint', 'bone'}
graph = ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2))

# bone
def gen_bone(dataset, set):
    print(dataset, set)
    data = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}/{}_bone.npy'.format(dataset, set),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    for v1, v2 in tqdm(graph):
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

# jmb
def merge_joint_bone_data(dataset, set):
    print(dataset, set)
    data_jpt = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set), mode='r')
    data_bone = open_memmap('./data/{}/{}_bone.npy'.format(dataset, set), mode='r')
    N, C, T, V, M = data_jpt.shape
    data_jpt_bone = open_memmap('./data/{}/{}_joint_bone.npy'.format(dataset, set), dtype='float32', mode='w+', shape=(N, 6, T, V, M))
    data_jpt_bone[:, :C, :, :, :] = data_jpt
    data_jpt_bone[:, C:, :, :, :] = data_bone

# motion  
def gen_motion(dataset, set,part):
    print(dataset, set, part)
    data = open_memmap('./data/{}/{}_{}.npy'.format(dataset, set, part),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}/{}_{}_motion.npy'.format(dataset, set, part),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
    fp_sp[:, :, T - 1, :, :] = 0

# angle ,From UAV-SAR
def gen_angle(dataset, set):
    print(dataset, set)
    data = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}/{}_angle.npy'.format(dataset, set),dtype='float32',mode='w+',shape=(N, 9, T, V, M))
    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i: fp_sp.__setitem__(i,getThridOrderRep(data[i])))(i) for i in tqdm(range(N)))

# reverse , From FR-AGCN
@numba.jit(nopython=True)
def reverse(data):
    C, T, V, M = data.shape
    arr = data.transpose(1,0,2,3) # C T V M to T C V M
    for i in range(T):
        if np.all(arr[i]==0):
            zidx = i
    zidx = T
    tmp = np.zeros((C, T, V, M))
    tmp[:, :zidx, :, :] = data[:, zidx-1::-1, :, :]
    return tmp

def gen_reverse(dataset, set):    
    print(dataset, set)
    data = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}/{}_joint_reverse.npy'.format(dataset, set),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i: fp_sp.__setitem__(i,reverse(data[i])))(i) for i in tqdm(range(N)))
    
if __name__ == '__main__':
    args = parser.parse_args()

    if args.modal == 'bone':   
        for dataset in datasets:
            for set in sets:
                gen_bone(dataset, set)
    elif args.modal == 'jmb':
        for dataset in datasets:
            for set in sets:
                merge_joint_bone_data(dataset, set)
    elif args.modal == 'motion':
        for dataset in datasets:
            for set in sets:
                for part in parts:
                    gen_motion(dataset, set, part)
    elif args.modal == 'angle':
        for dataset in datasets:
            for set in sets:
                gen_angle(dataset, set)
    elif args.modal == 'reverse':
        for dataset in datasets:
            for set in sets:
                gen_reverse(dataset, set)
    else:
        raise ValueError('Invalid Modal')
