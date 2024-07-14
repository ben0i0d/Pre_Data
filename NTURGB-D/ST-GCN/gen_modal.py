import argparse
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

parser = argparse.ArgumentParser(description='Dataset Preprocessing')
parser.add_argument('--modal', type=str, default='bone', help='gen_modal')

paris = {
    'xview': (
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
        (12, 0), 
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (20, 20),(22, 7), (23, 24), 
        (24, 11)
    ),
    'xsub': (
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
        (12, 0), 
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (20, 20),(22, 7), (23, 24), 
        (24, 11)
    )
}

sets = {
    'train', 'val'
}

datasets = {
    'xsub', 'xview'
}

parts = {
    'joint', 'bone'
}

# bone
def gen_bone(dataset, set):
    print(dataset, set)
    data = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = np.zeros((N, 3, T, V, M), dtype='float32')
    #fp_sp = open_memmap('./data/{}/{}_joint_bone.npy'.format(dataset, set), dtype='float32', mode='w+', shape=(N, 3, T, V, M))
    for v1, v2 in tqdm(paris[dataset]):
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
    np.save('./data/{}/{}_bone.npy'.format(dataset, set), fp_sp)

# jmb
def merge_joint_bone_data(dataset, set):
    print(dataset, set)
    data_jpt = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set), mode='r')
    data_bone = open_memmap('./data/{}/{}_bone.npy'.format(dataset, set), mode='r')
    N, C, T, V, M = data_jpt.shape
    data_jpt_bone = open_memmap('./data/{}/{}_joint_bone.npy'.format(dataset, set), dtype='float32', mode='w+', shape=(N, 6, T, V, M))
    data_jpt_bone[:, :C, :, :, :] = data_jpt
    data_jpt_bone[:, C:, :, :, :] = data_bone

def gen_motion(dataset, set,part):
    print(dataset, set, part)
    data = open_memmap('./data/{}/{}_{}.npy'.format(dataset, set, part),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = np.zeros((N, 3, T, V, M), dtype='float32')
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
    fp_sp[:, :, T - 1, :, :] = 0
    np.save('./data/{}/{}_{}_motion.npy'.format(dataset, set, part), fp_sp)

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
    else:
        raise ValueError('Invalid Modal')
