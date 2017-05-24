'''
This code is combine prevlous code to do the freature extraction
i cut the sample to the same length 30000
win_len = 1024,over_lap = 25%,per sample obtain 40 frames and per frame obtain 187 dims features 
'''
"""
Created on 5/8/2017

@author: panzhanpeng
"""
import glob
import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from skimage import transform,io

# matplotlib inline
try:
    import cPickle as pickle  # Improve speed
except ValueError:
    import pickle

win_len = 256  # window length of per sample
hop_len = win_len / 2  # overlap = 50%
sr = 50000  # sample rate

def extract_feature(file_name, sr, win_len, hop_len):
    input_image = np.empty([235, 129, 3], np.uint8)
    X, sample_rate = librosa.load(file_name, sr)  # load the wav file
    X = X[0:30000]  # cut the X to the same length(30000)
    stft = np.abs(librosa.stft(X, n_fft=win_len, hop_length=hop_len, win_length=win_len)).T
    # pcastft = pca.fit_transform(stft)
    scalestft = preprocessing.minmax_scale(stft, [0, 255])
    intstft = np.uint8(scalestft)
    input_image[:, :, 0] = intstft
    input_image[:, :, 1] = intstft
    input_image[:, :, 2] = intstft
    output = transform.resize(input_image, (224, 224))
    return output


def parse_audio_files(parent_dir, sub_dirs, sr, win_len, hop_len, dir, file_ext='*.wav'):
    labels,image =  np.empty(0,np.uint8), np.empty([0,224*224*3],np.uint8)
    i = 0
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):

            stft = extract_feature(fn, sr, win_len, hop_len)
            i = i + 1
            if 'N' in sub_dir:
                io.imsave(dir+'N'+str(i)+'.jpeg', stft)
            if 'P' in sub_dir:
                io.imsave(dir +'P'+ str(i)+'.jpeg', stft)

parent_dir = '/home/pzp/data/Saarbruecken Voice Database/2000data'
tr_sub_dirs = ['Pathology/Train', 'Normal/Train']
ts_sub_dirs = ['Pathology/Test', 'Normal/Test']
tr_data = '/home/pzp/data/train/'
ts_data = '/home/pzp/data/test/'
parse_audio_files(parent_dir, tr_sub_dirs, sr, win_len, hop_len, tr_data, file_ext='*.wav')
parse_audio_files(parent_dir, ts_sub_dirs, sr, win_len, hop_len, ts_data, file_ext='*.wav')

