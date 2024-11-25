import os
import psutil
import numpy as np
from tqdm import tqdm
from scipy.stats import mode
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

def gen_data(dataset, modal, file_list, shape):
    #test不存在多模态
    if dataset == 'test':
        modal = ''
    #生成实际file位置
    file_list = list(file_list)
    for i in range(len(file_list)):
        file_list[i] = os.path.join('data/raw_data',dataset,modal,file_list[i])
    #生成数据
    tmp = open_memmap('data/npy_data/tmp.npy'.format(dataset, modal),dtype=np.float32,mode='w+',shape=(shape[1], shape[0], shape[2]))
    
    if dataset == 'test':
        data = open_memmap('data/npy_data/{}/test.npy'.format(dataset),dtype=np.float32,mode='w+',shape=shape)
    else:
        data = open_memmap('data/npy_data/{}/{}.npy'.format(dataset, modal),dtype=np.float32,mode='w+',shape=shape)
        
    if dataset == 'train':
        Parallel(n_jobs=psutil.cpu_count(), verbose=0)(delayed(lambda i: tmp.__setitem__(i, np.loadtxt(file_list[i])))(i) for i in tqdm(range(9)))
    else:
        Parallel(n_jobs=psutil.cpu_count(), verbose=0)(delayed(lambda i: tmp.__setitem__(i, np.loadtxt(file_list[i],delimiter=',')))(i) for i in tqdm(range(9)))
    
    data = tmp.transpose(1,0,2)
    os.remove('data/npy_data/tmp.npy')
    
    # process some nan
    if dataset == 'train' and modal == 'Hips':
        data[121217] = np.nan_to_num(data[121217],copy=False,nan=0.005)

def gen_label(set, modal):
    source_path = os.path.join('data/raw_data',set,modal,'Label.txt')

    source = np.loadtxt(source_path, dtype=np.int8)
    label = np.zeros((len(source)),dtype=np.int8)
    
    Parallel(n_jobs=psutil.cpu_count(), verbose=0)(delayed(lambda i: label.__setitem__(i, mode(source[i])[0]))(i) for i in tqdm(range(len(label))))
    
    np.save('data/npy_data/{}/label.npy'.format(set),label)

if __name__ == '__main__':

    #形状对应：帧，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，500个样本
    shape_list = [(196072, 9, 500),(28789, 9, 500),(92726, 9, 500)]

    sets = ('train', 'valid','test')

    modal_list = ("Bag","Hand","Hips","Torso")

    file_list = ("Acc_x.txt", "Acc_y.txt", "Acc_z.txt","Gyr_x.txt", "Gyr_y.txt", "Gyr_z.txt","Mag_x.txt", "Mag_y.txt", "Mag_z.txt")

    if not os.path.exists('data/npy_data'):
        os.mkdir('data/npy_data')
        for dataset in sets:
            path=os.path.join('data/npy_data',dataset)
            if not os.path.exists(path):
                os.mkdir(path)

    # gen_label
    for set in sets:
        if set != 'test':
            print("processing {} Label".format(set))
            gen_label(set,modal_list[0])

    # gen_data
    for set in sets:
        for modal in modal_list:
            if set == 'test' and modal == 'Bag' :
                print("processing {}".format(set))
                gen_data(set, modal_list[0],file_list,shape_list[sets.index(set)])
            elif set == 'train' or set == 'valid':
                print("processing {} {}".format(set,modal))
                gen_data(set, modal,file_list,shape_list[sets.index(set)])