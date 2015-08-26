########################################################
# CS.py: compressive sensing based estimation
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/21
# Implemented approaches: CS [Luo et al., MobiCom'2009]
########################################################

import numpy as np 
from numpy import linalg as LA
import time, sys
import random
import commons
from scipy.fftpack import dct
import pywt
from sklearn.linear_model import Lasso


#======================================================#
# Sampling to generate trainMatrix, observedMatrix, testMatrix
#======================================================#
def sampling(matrix, rate, roundId, para):
    trainingPeriod = para['trainingPeriod']
    trainMatrix = matrix[:, 0:trainingPeriod]
    testMatrix = matrix[:, trainingPeriod:] 
    observedMatrix = commons.removeEntries(testMatrix, rate, roundId)
    return trainMatrix, observedMatrix, testMatrix


#======================================================#
# Function to recover the unobserved values
#======================================================#
def recover(trainMatrix, observedMatrix, para):
    (numNode, numTime) = observedMatrix.shape
    avgVec = np.average(trainMatrix, axis=1)
    recoveredMatrix = np.zeros((numNode, numTime))
    transformMatrix = transformBasis(numNode, para)
    for i in xrange(numTime):
        # transform
        (monitors, ) = np.nonzero(observedMatrix[:, i] > 0)
        observedVec = observedMatrix[monitors, i]
        measureMatrix = transformMatrix[monitors, :]
        # lasso recovery
        lasso = Lasso(alpha=para['lmbda'])
        lasso.fit(measureMatrix, observedVec - avgVec[monitors])
        recoveredMatrix[:, i] = np.dot(transformMatrix, lasso.coef_) + avgVec
    recoveredMatrix[observedMatrix > 0] = observedMatrix[observedMatrix > 0]
    return recoveredMatrix


#======================================================#
# Function to get transform basis
#======================================================#
def transformBasis(numNode, para):
    if para['transform'] == 'DCT':
        transformMatrix = dct(np.eye(numNode), type=3, axis=0, norm='ortho')
        transformMatrix = transformMatrix.T # note inverse == transpose
    elif 'DWT' in para['transform']:
        wt = para['transform'].split('-')[1]
        dwtMaxLevel = pywt.dwt_max_level(numNode, filter_len=pywt.Wavelet(wt).dec_len)
        eyeMatrix = np.eye(numNode)
        transformMatrix = []
        for i in xrange(numNode):
            coeffs = pywt.wavedec(eyeMatrix[:, i], wt, level=dwtMaxLevel) 
            transformMatrix.append(np.hstack(coeffs))
        transformMatrix = np.vstack(transformMatrix).T
        transformMatrix = LA.pinv(transformMatrix)
    else:
        print 'para[\'transform\'] error!'
        sys.exit() 
    return transformMatrix

