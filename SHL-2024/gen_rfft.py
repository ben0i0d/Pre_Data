import os
import psutil
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

def rfft(tmp):
    data = [[] for _ in tmp]
    for i in range(len(tmp)):
        data[i] = np.abs(np.fft.rfft(tmp[i]-tmp[i].mean()))
        data[i] /= (500 / 2)
        data[i][0] /= 2
    return data

def gen_rfft(dataset, shape):
    source = np.load('data/npy_data/{}/data.npy'.format(dataset),mmap_mode='r')
    data = open_memmap('data/fft_data/{}/data.npy'.format(dataset),dtype=np.float32,mode='w+',shape=shape)
    for i in tqdm(range(shape[0])):
        Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda j: data[i].__setitem__(j,rfft(source[i,j])))(j) for j in range(shape[1]))

def gen_test(shape=(9, 92726, 251)):
    if not os.path.exists('data/fft_data/test'):
        os.mkdir('data/fft_data/test')
    #生成数据
    source = np.load('data/npy_data/test/data.npy',mmap_mode='r')
    data = open_memmap('data/fft_data/test/data.npy',dtype=np.float32,mode='w+',shape=shape)

    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i: data.__setitem__(i,rfft(source[i])))(i) for i in tqdm(range(shape[0])))

if __name__ == '__main__':

    #形状对应：模态，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，帧，500个样本
    shape_list = [(4, 9, 196072, 251),(4, 9, 28789, 251)]

    sets = ('train', 'valid')

    modal_list = ("Bag","Hand","Hips","Torso")

    if not os.path.exists('data/fft_data'):
        os.mkdir('data/fft_data')
        for set in sets:
            path=os.path.join('data/fft_data',set)
            if not os.path.exists(path):
                os.mkdir(path)
    
    # copy label
    for set in sets:
        print("processing {} Label".format(set))
        copyfile('data/npy_data/{}/label.npy'.format(set),'data/fft_data/{}/label.npy'.format(set))

    # gen test
    print("processing test data")
    gen_test()

    # gen_rfft
    for i,set in enumerate(sets):
        print("processing {}".format(set))
        gen_rfft(set, shape_list[i])
