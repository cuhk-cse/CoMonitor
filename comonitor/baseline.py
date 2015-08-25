########################################################
# CS.py: compressive sensing based estimation
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/21
########################################################

import numpy as np 
from numpy import linalg as LA
import time
import random
from commons.util import logger
from scipy.fftpack import dct
import pywt
from sklearn.linear_model import Lasso 
from matplotlib.pyplot import plot, show, figure, title

#======================================================#
# Function to recover the unobserved values
#======================================================#
def recover(matrix, trainMatrix, para):
    (numNode, numTime) = trainMatrix.shape
    recoveredMatrix = np.zeros((numNode, numTime))
    transformMatrix = transformBasis(numNode, para)
    for i in xrange(numTime):
        # transform
        (monitors, ) = np.nonzero(trainMatrix[:, i] > 0)
        observedVec = trainMatrix[monitors, i]
        measureMatrix = transformMatrix[monitors, :]
        # lasso recovery
        lasso = Lasso(alpha=para['lmbda'])
        lasso.fit(measureMatrix, observedVec)
        # plot(lasso.coef_, 'x')
        # show()
        recoveredMatrix[:, i] = reverseTransform(transformMatrix, lasso.coef_, para)
    recoveredMatrix[trainMatrix > 0] = trainMatrix[trainMatrix > 0]
    return recoveredMatrix


#======================================================#
# Function to get transform basis
#======================================================#
def transformBasis(numNode, para):
    if para['transform'] == 'DCT':
        transformMatrix = dct(np.eye(numNode), type=3, axis=0, norm='ortho')
        transformMatrix = transformMatrix.T # note inverse == transpose
    # elif para['transform'] == 'DWT':
        # transformMatrix = fft(np.eye(numNode))
        # transformMatrix = LA.inv(transformMatrix)
    return transformMatrix


#======================================================#
# Function to get recovered vector
#======================================================#
def reverseTransform(transformMatrix, recoveredCoef, para):
    if para['transform'] == 'DCT':
        recoveredVec = np.dot(transformMatrix, recoveredCoef)
    # elif para['transform'] == 'DWT':
        # recoveredVec = np.dot(transformMatrix, recoveredCoef).real
    return recoveredVec

