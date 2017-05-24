# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:44:30 2016

@author: panzhanpeng
"""
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA 
from sklearn import preprocessing

win_len = 256                 #window length of per sample
hop_len = win_len/2              #overlap = win_len - hop_length
sr = 50000
X, sample_rate = librosa.load('123.wav',sr)
X = X[len(X)-30000:len(X)+1]
stft = np.abs(librosa.stft(X,n_fft = win_len,hop_length=hop_len,win_length=win_len)).T
stft1 = librosa.stft(X,n_fft = win_len,hop_length=hop_len,win_length=win_len)
ext_features = preprocessing.minmax_scale(stft,axis=1)
mel_s = librosa.logamplitude(librosa.feature.melspectrogram(X, sr=sample_rate,n_fft=win_len,hop_length=hop_len))
mel_log = librosa.power_to_db(librosa.feature.melspectrogram(X, sr=sample_rate,n_fft=win_len,hop_length=hop_len))
#mfccs = librosa.feature.mfcc(S=mel_s, sr=sample_rate, n_mfcc=40).T
#chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T
#mel = librosa.feature.melspectrogram(X, sr=sample_rate,n_fft=win_len,hop_length=hop_len).T
#contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T
#tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T
pca=PCA(n_components='mel')
#mfccs = pca.fit_transform(mfccs)
chroma = pca.fit_transform(chroma)
mel = pca.fit_transform(mel)
contrast = pca.fit_transform(contrast)
'''
features = np.empty((0,40,187))
ext_features = np.hstack([mfccs,chroma,mel,contrast])
ext_features = preprocessing.scale(ext_features,axis=1)
pca=PCA(   n_components=30)
pca_tr=pca.fit_transform(ext_features)
ext_features = ext_features[np.newaxis,:,:]
features = np.vstack([features,ext_features])

'''



