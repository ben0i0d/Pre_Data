import argparse
from tqdm import tqdm
from numpy.lib.format import open_memmap

parser = argparse.ArgumentParser(description='Dataset Preprocessing')
parser.add_argument('--modal', type=str, default='bone', help='gen_modal')

# ucla graph
# [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10),(12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
# remeber to -1 for python index

sets = {'train', 'val'}
parts = {'joint', 'bone'}
graph = [(0, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18)]


# bone
def gen_bone(set):
    print(set)
    data = open_memmap('./data/npy/{}_joint.npy'.format(set),mode='r')
    N, C, T, V = data.shape
    fp_sp = open_memmap('./data/npy/{}_bone.npy'.format(set),dtype='float32',mode='w+',shape=(N, 3, T, V))
    for v1, v2 in tqdm(graph):
        fp_sp[:, :, :, v1] = data[:, :, :, v1] - data[:, :, :, v2]

# jmb
def merge_joint_bone_data(set):
    print(set)
    data_jpt = open_memmap('./data/npy/{}_joint.npy'.format(set), mode='r')
    data_bone = open_memmap('./data/npy/{}_bone.npy'.format(set), mode='r')
    N, C, T, V = data_jpt.shape
    data_jpt_bone = open_memmap('./data/npy/{}_joint_bone.npy'.format(set), dtype='float32', mode='w+', shape=(N, 6, T, V))
    data_jpt_bone[:, :C, :, :] = data_jpt
    data_jpt_bone[:, C:, :, :] = data_bone

# motion  
def gen_motion(set,part):
    print(set, part)
    data = open_memmap('./data/npy/{}_{}.npy'.format(set, part),mode='r')
    N, C, T, V = data.shape
    fp_sp = open_memmap('./data/npy/{}_{}_motion.npy'.format(set, part),dtype='float32',mode='w+',shape=(N, 3, T, V))
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :] = data[:, :, t + 1, :] - data[:, :, t, :]
    fp_sp[:, :, T - 1, :] = 0

if __name__ == '__main__':
    args = parser.parse_args()

    if args.modal == 'bone':   
        for set in sets:
            gen_bone(set)
    elif args.modal == 'jmb':
            for set in sets:
                merge_joint_bone_data(set)
    elif args.modal == 'motion':
            for set in sets:
                for part in parts:
                    gen_motion(set, part)
    else:
        raise ValueError('Invalid Modal')
