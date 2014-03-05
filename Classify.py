import numpy as np
import matplotlib.pyplot as plt
import wave
import struct
import sys
import random
import operator
import scipy
import math
from pylab import*
import cmath


def accuracy(index):
	count1 = 0
	for i in range(0,50):
		if np.ceil(index[i]/50) == 0:
			count1 = count1 + 1
	print count1
	print "Accuracy for Dhol is",(count1/50.0)*100

	count2 = 0
	for i in range(50,100):
		if np.ceil(index[i]/50) == 1:
			count2 = count2 + 1
	print count2
	print "Accuracy for violin is",(count2/50.0)*100

	count3 = 0
	for i in range(100,150):		
		if np.ceil(index[i]/50) == 2:
			count3 = count3 + 1
	print count3
	print "Accuracy for flute is",(count3/50.0)*100

	count4 = 0
	for i in range(150,200):
		if np.ceil(index[i]/50) == 3:
			count4 = count4 + 1
	print count4
	print "Accuracy for piano is",(count4/50.0)*100



train = np.load('train_new.npy')
print shape(train)
array = (train - train.min())/(train.max()-train.min())#normalise
#subplot(211)
#title('Train dataset MFCC')
#im = imshow(train,aspect='auto',interpolation='nearest')

test = np.load('test_new.npy')
print shape(test)
test = (test - test.min())/(test.max()-test.min())#normalise
#subplot(212)
#title('Test dataset MFCC')
#im = imshow(test,aspect='auto',interpolation='nearest')
#show()

total = np.dot(train.T,test)
print shape(total)
index = total.argmax(axis=1)
print shape(index)


#Evaluate accuracy
accuracy(index)



#accuracy of self similarity matrix
print "self accuracy"
total = np.dot(train.T,train)
total = (total - total*np.eye(200, dtype=int))
im = imshow(total,aspect='auto',interpolation='nearest')
show()
index = total.argmax(axis=1)
accuracy(index)



print 'end'

