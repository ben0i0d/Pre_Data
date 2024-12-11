import os
import psutil
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap

def window_rfft(tmp, window_length=100, step=20, fs=100):
    """
    Computes the RFFT for segments of signals using a sliding window.
    """
    data = [[] for _ in tmp]
    for i,signal in enumerate(tmp):
        signal_data = []
        for start in range(0, len(signal) - window_length + 1, step):
            window = signal[start:start + window_length]
            window = window - np.mean(window)
            rfft_result = np.abs(np.fft.rfft(window))
            rfft_result /= (fs / 2)
            rfft_result[0] /= 2
            signal_data.append(rfft_result)
        data[i] = np.array(signal_data).reshape(-1)

    return data

def gen_rfft(dataset, shape):
    source = np.load('data/npy_data/{}/data.npy'.format(dataset),mmap_mode='r')
    data = open_memmap('data/fft_data/{}/data.npy'.format(dataset),dtype=np.float32,mode='w+',shape=shape)
    for i in tqdm(range(shape[0])):
        Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda j: data[i].__setitem__(j,window_rfft(source[i,j])))(j) for j in range(shape[1]))

def gen_test(shape=(9, 92726, 1071)):
    if not os.path.exists('data/fft_data/test'):
        os.mkdir('data/fft_data/test')
    #生成数据
    source = np.load('data/npy_data/test/data.npy',mmap_mode='r')
    data = open_memmap('data/fft_data/test/data.npy',dtype=np.float32,mode='w+',shape=shape)

    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i: data.__setitem__(i,window_rfft(source[i])))(i) for i in tqdm(range(shape[0])))

if __name__ == '__main__':

    #形状对应：模态，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，帧，500个样本
    shape_list = [(4, 9, 196072, 1071),(4, 9, 28789, 1071)]

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
