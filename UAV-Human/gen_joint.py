"""
Script to process raw data and generate dataset's binary files:
    - .npy skeleton data files: np.array of shape B x C x V x T x M
    - .npy label files: (label: list[int])
"""
import os
import re
import glob
import numba
import psutil
import numpy as np
from tqdm import tqdm
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

from preprocess import pre_normalization

MAX_BODY_TRUE = 2
MAX_BODY_KINECT = 4
NUM_JOINT = 17
MAX_FRAME = 300

def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence

@numba.jit(nopython=True)
def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std() # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body, num_joint):
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:MAX_BODY_TRUE]
    data = data[index, :MAX_FRAME, :, :]

    # pad to MAX_FRAME
    data = np.pad(data, ((0, 0), (0, MAX_FRAME - data.shape[1]), (0, 0), (0, 0)), 'constant', constant_values=0)
    # pad the null frames with the previous frames
    for i_p, person in enumerate(data):
        if person.sum() == 0:
            continue
        if person[0].sum() == 0:
            index = (person.sum(-1).sum(-1) != 0)
            tmp = person[index].copy()
            person *= 0
            person[:len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    data[i_p, i_f:] = pad
                    break

    return data.transpose(3, 1, 2, 0) # M,T,V,C To C,T,V,M


def gendata(data_path,split):

    out_path = data_path
    data_path = os.path.join(data_path, split)

    skeleton_filenames = [os.path.basename(f) for f in glob.glob(os.path.join(data_path, "**.txt"), recursive=True)]

    FILENAME_REGEX = r'P\d+S\d+G\d+B\d+H\d+UC\d+LC\d+A(\d+)R\d+_\d+'
    label = Parallel(n_jobs=psutil.cpu_count(logical=False), backend='threading', verbose=0)(delayed(lambda i: int(re.match(FILENAME_REGEX, i).groups()[0]))(i) for i in skeleton_filenames)
    np.save('{}/{}_label.npy'.format(out_path, split), label)

    sample_name = Parallel(n_jobs=psutil.cpu_count(logical=False), backend='threading', verbose=0)(delayed(lambda i: os.path.join(data_path, i))(i) for i in skeleton_filenames)

    data = open_memmap('{}/{}_joint.npy'.format(out_path, split),dtype='float32',mode='w+',shape=((len(sample_name), 3, MAX_FRAME, NUM_JOINT, MAX_BODY_TRUE)))
    
    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i,s: data.__setitem__(i,read_xyz(s, max_body=MAX_BODY_KINECT, num_joint=NUM_JOINT)))(i,s) for i,s in enumerate(tqdm(sample_name)))

    # check no skeleton
    for i in range(len(data)):
        if np.all(data[i,:] == 0):
            print("{} {} has no skeleton".format(data_path, i))

    data = data.transpose(0, 4, 2, 3, 1)  # N, C, T, V, M  to  N, M, T, V, C

    # Center the human at origin
    # sub the center joint #1 (spine joint in ntu and neck joint in kinetics)'
    for i_s, skeleton in enumerate(data):
        if skeleton.sum() == 0:
            continue
        # skeleton[0][:, center_joint:center_joint+1, :].copy() | uav center_joint=1
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            # reshape(T, V, 1) | uav T=300 v=17
            mask = (person.sum(-1) != 0).reshape(300, 17, 1)
            data[i_s, i_p] = (data[i_s, i_p] - main_body_center) * mask

    data = pre_normalization(data).transpose(0, 4, 2, 3, 1) # N, M, T, V, C to N, C, T, V, M
    
if __name__ == '__main__':

    path_list = ['data/v1','data/v2']
    part = ['train','test']
    
    processes = []
    for path in path_list:
        for p in part:
            gendata(path, p)