import os
import glob
import psutil
import pickle
import numpy as np
from tqdm import tqdm
import audioflux as af
from joblib import Parallel , delayed

# 进入data目录
os.chdir('./data')

# 定义snr目录列表和device目录列表
snrs = ['0dB', '6dB', '-6dB']
devices = ['pump', 'slider', 'valve', 'fan']
ids = ['id_00', 'id_02', 'id_04', 'id_06']
labels = ['normal', 'abnormal']

# 定义映射字典
snrm = {'0dB': 0, '6dB': 1, '-6dB': -1}
devicem = {'fan': 1, 'pump': 2, 'slider': 3, 'valve': 4}
labelm = {'normal': 1, 'abnormal': 0}

def mfcc_feature(file_name):
    audio, sr = af.read(file_name)
    mfcc, _ = af.mfcc(audio, samplate=sr)
    return mfcc.reshape(-1)

data = []

# 遍历目录
for snr in snrs:
    for device in devices:
        print("processing {} {}".format(snr, device))
        for did in ids:
            for label in labels:
                wav_files = [os.path.basename(f) for f in glob.glob(os.path.join(snr, device, did, label, "**.wav"), recursive=True)]
                mfccs = Parallel(n_jobs=psutil.cpu_count(logical=True), verbose=0)(delayed(lambda i: mfcc_feature(os.path.join(snr, device, did, label, i)))(i) for i in tqdm(wav_files))

                for mfcc in mfccs:
                    data.append([mfcc, snrm[snr], devicem[device], labelm[label]])

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)