import os
import psutil
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

def rfft(tmp):
    data = np.zeros((9,251))
    for i in range(9):
        data[i] = np.abs(np.fft.rfft(tmp[i]-tmp[i].mean()))
        data[i] /= (500 / 2)
        data[i][0] /= 2
    return data

def gen_rfft(dataset, modal, shape):
    if dataset == 'test':
        source = np.load('data/npy_data/{}/test.npy'.format(dataset),mmap_mode='r')
        data = open_memmap('data/fft_data/{}/test.npy'.format(dataset),dtype=np.float32,mode='w+',shape=shape)
    else:
        source = np.load('data/npy_data/{}/{}.npy'.format(dataset, modal),mmap_mode='r')
        data = open_memmap('data/fft_data/{}/{}.npy'.format(dataset, modal),dtype=np.float32,mode='w+',shape=shape)

    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i: data.__setitem__(i,rfft(source[i])))(i) for i in tqdm(range(shape[0])))
    
if __name__ == '__main__':

    #形状对应：帧，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，500个样本
    shape_fft_list = [(196072, 9, 251),(28789, 9, 251),(92726, 9, 251)]

    sets = ('train', 'valid','test')

    modal_list = ("Bag","Hand","Hips","Torso")

    if not os.path.exists('data/fft_data'):
        os.mkdir('data/fft_data')
        for dataset in sets:
            path=os.path.join('data/fft_data',dataset)
            if not os.path.exists(path):
                os.mkdir(path)
    
    for set in sets:
        if set != 'test':
            copyfile('data/npy_data/{}/label.npy'.format(set),'data/fft_data/{}/label.npy'.format(set))

    for set in sets:
        for modal in modal_list:
            if set == 'test' and modal == 'Bag' :
                print("processing {} {}".format(set,modal))
                gen_rfft(set, modal,shape_fft_list[sets.index(set)])
            elif set == 'train' or set == 'valid':
                print("processing {} {}".format(set,modal))
                gen_rfft(set, modal,shape_fft_list[sets.index(set)])
