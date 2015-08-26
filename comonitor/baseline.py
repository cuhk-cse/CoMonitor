########################################################
# baseline.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/26
# Implemented approach: baseline approach
########################################################

import numpy as np 
from numpy import linalg as LA
import time
import commons


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
    (numNode, numTestTime) = observedMatrix.shape
    avgTrain = np.average(trainMatrix, axis=1)
    recoveredMatrix = np.zeros((numNode, numTestTime))
    for i in xrange(numTestTime):
        if i == 0:
            recoveredMatrix[:, 0] = trainMatrix[:, -1]
        else:
            recoveredMatrix[:, i] = observedMatrix[:, i - 1]
            (monitors, ) = np.nonzero(observedMatrix[:, i] > 0)
            observedVec = observedMatrix[monitors, i]
            avgVec = (np.average(observedVec) + avgTrain) / 2
            idx = (recoveredMatrix[:, i] == 0)
            recoveredMatrix[idx, i] = avgVec[idx]
    recoveredMatrix[observedMatrix > 0] = observedMatrix[observedMatrix > 0]
    return recoveredMatrix

