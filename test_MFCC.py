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
		if area==0:
			area=1
		y2 = (y/area)
		bank[j,binrep[j]:binrep[j+2]] = y2
	im = imshow(np.flipud(bank),aspect='auto', interpolation='nearest')
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





"""
Read input audio wav file and calculate spectrogram
"""
p, f, sound_info = read_audio('violin1.wav')
print "frequency is",f
print len(sound_info)


"""
Calculate spectrogram
"""

Pxx, freqs, bins, im = spectrogram(sound_info, f, 4096, 256)
print "shape of Spectrogram is",shape(Pxx)


"""
Generate filterabank
"""
M = generate_filterbank(channels=40,NFFT=4096,fs=f,fmin=0,fmax=20000)
print "shape of filterabank is",shape(M)



"""
Generate Mel spectrogram as the dot product of spectrogram and mel filterabank
"""
subplot(3,1,2)
title('Mel spectrogram')
#to get rid of all zeroes in matrix
sal = np.dot(M,Pxx)
for i in range(0,len(sal)):
	for j in range(0,len(sal[0])):
		if sal[i][j] == 0:
			sal[i][j]+=1

sal = 10*np.log10(sal)
print "shape of Mel Spectrogram is",shape(sal)
im = imshow(sal,aspect='auto',interpolation='nearest',origin='lower')


"""
Generate DCT matrix
"""
D = dctmtx(40)
D = np.flipud(D)
print "shape of DCT is",shape(D)


"""
Generate MFCC matrix as dot product of Mel Spectrum and DCT matrix
"""
subplot(3,1,3)
title('MFCC')
MFCC = np.dot(D[1:15,:],sal)
im = imshow(MFCC,aspect='auto',interpolation='nearest',origin='lower')


#Just for plotting
subplot(3,1,1)
title('Spectrogram')
Pxx, freqs, bins, im = spectrogram(sound_info, f, 4096, 512)
plot(Pxx,freqs)
ylim(0,22050)
xlim(0,(len(sound_info)/44100.00))
show()



print "end"


