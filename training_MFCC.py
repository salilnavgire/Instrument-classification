import numpy as np
import matplotlib.pyplot as plt
import wave
import struct
import sys
from scikits.audiolab import *
import random
from datetime import datetime
import operator
import scipy
import math
from pylab import*
import cmath
from tempfile import TemporaryFile

def hz_to_mel(x):
	val = 1127.01028*np.log(1+x/700.0)
	return val


def mel_to_hz(x):
	val = 700.0*(cmath.e**(x/1127.01028)-1.0)
	return val


def generate_filterbank(channels,NFFT,fs,fmin,fmax):
	melmin = hz_to_mel(fmin)
	melmax = hz_to_mel(fmax)

	melpts = np.linspace(melmin,melmax,channels+2)
	bin = mel_to_hz(melmin+np.arange((channels+2),dtype=np.double)/(channels+1)*(melmax - melmin))
	binrep = np.floor((NFFT+2)*bin/(2*((fs/2)-fmin)))
	bank = np.zeros([channels,NFFT/2+1],dtype=float)

	for j in xrange(0,channels):
		y = np.hamming(binrep[j+2]-binrep[j])
		area = trapz(y, dx=5)
		y2 = (y/area)*10
		bank[j,binrep[j]:binrep[j+2]] = y2
	return bank


def dctmtx(n):
    row,col = meshgrid(range(n), range(n))
    dct=sqrt(2.0/n)*cos(pi*(2*row-1)*col/(2*n))
    dct[0]/=sqrt(2)
    return dct


def spectrogram(sound_info,f,nfft,hop):
	Pxx, freqs, bins, im = specgram(sound_info, Fs = f, NFFT = nfft, noverlap=nfft-hop, scale_by_freq=True,sides='default')
	return Pxx, freqs, bins, im


def read_audio(filename):
	spf = wave.open(filename,'r')
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	p = spf.getnchannels()
	f = spf.getframerate()
	sound_info = np.zeros(len(signal),dtype=float)
	signal = signal.astype(np.float)
	sound_info = signal/max(signal)
	return p ,f , sound_info


def generate_MFCCvector(filenm):
	p, f, sound_info = read_audio(filenm)
	sound_info = sound_info[1:len(sound_info):2]#all my signals were stereo
	print "frequency is",f
	print 'min',sound_info.min()
	print 'max',sound_info.max()
	sound_info = np.absolute(sound_info)
	print 'min',sound_info.min()
	print 'max',sound_info.max()
	print "length of audio",len(sound_info)
	count = 0

	new_sound=list()

	#rms thresholding
	for i in range(len(sound_info)):
		if sound_info[i]**2>0.00010:
			count+=1
			new_sound.append(sound_info[i])

	print "new audio length",len(new_sound)
	print "percent good",count*100/len(sound_info)
	Pxx, freqs, bins, im = spectrogram(new_sound, f, 2048,512)
	print "shape of spectrum",shape(Pxx)
	
	M = generate_filterbank(channels=40,NFFT=2048,fs=f,fmin=0,fmax=5000)
	print "shape of filterbank",shape(M)

	sal = np.dot(M,Pxx)
	sal = 10*np.log10(sal)
	print "shape of Mel spectrum",shape(sal)

	D = dctmtx(40)
	D = D[1:15,:]
	print "shape of DCT",shape(D)

	MFCC = np.dot(D,sal)
	print "shape of MFCC",shape(MFCC)
	MFCC = MFCC[:,0:8600]
	print "shape of MFCC",shape(MFCC)

	grand = MFCC.reshape(-1,86).mean(axis=1).reshape(MFCC.shape[0],-1)
	print shape(grand)
	return grand




print "dhol"
title('Dhol')
grand1 = generate_MFCCvector('dhol.wav')
print shape(grand1)

print "violin"
title('violin')
grand2 = generate_MFCCvector('violin_sample.wav')
print shape(grand2)

print 'flute'
title('flute')
grand3 = generate_MFCCvector('flute.wav')
print shape(grand3)

print 'piano'
title('piano')
grand4 = generate_MFCCvector('piano.wav')
print shape(grand4)

title('Final Training Dataset')
grand_final_train = np.hstack((grand1[:,0:50],grand2[:,0:50],grand3[:,0:50],grand4[:,0:50]))
train = TemporaryFile()
np.save('train.npy',grand_final_train)
print shape(grand_final_train)
im = imshow(grand_final_train,aspect='auto',interpolation='nearest')
show()

title('Final Test Dataset')
grand_final_test = np.hstack((grand1[:,50:100],grand2[:,50:100],grand3[:,50:100],grand4[:,50:100]))
test = TemporaryFile()
np.save('test.npy',grand_final_test)
print shape(grand_final_test)
im = imshow(grand_final_test,aspect='auto',interpolation='nearest')
show()


print "end"




