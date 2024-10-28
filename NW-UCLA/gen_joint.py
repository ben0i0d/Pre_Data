import os
import glob
import json
import numpy as np
from tqdm import tqdm

MAX_FRAME = 250 # real max frame = 201

def readjson(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['file_name'], np.array(data['skeletons']),  data['label']

if __name__ == '__main__':
    
    train_data = []
    val_data = []
    train_label = []
    val_label = []
    
    train_list = np.load('data/train_list.npy')
    val_list = np.load('data/val_list.npy')

    path = 'data/all_sqe'
    if not os.path.exists('data/npy'):
        os.makedirs('data/npy')

    skeleton_filenames = [os.path.basename(f) for f in glob.glob(os.path.join(path, "**.json"), recursive=True)]
    
    for skeleton_filename in tqdm(skeleton_filenames):
        json_path = os.path.join(path, skeleton_filename)
        file_name, skeletons, label = readjson(json_path)
        skeletons = np.pad(skeletons, ((0, MAX_FRAME - skeletons.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
        if file_name in train_list:
            train_data.append(skeletons)
            train_label.append(label)
        elif file_name in val_list:
            val_data.append(skeletons)
            val_label.append(label)

    train_data = np.array(train_data).transpose(0, 3, 1, 2) # N T V C to N C T V
    val_data = np.array(val_data).transpose(0, 3, 1, 2) # N T V C to N C T V
    train_label = np.array(train_label)
    val_label = np.array(val_label)

    np.save('data/npy/train_joint.npy', train_data)
    np.save('data/npy/val_joint.npy', val_data)
    np.save('data/npy/train_label.npy', train_label)
    np.save('data/npy/val_label.npy', val_label)