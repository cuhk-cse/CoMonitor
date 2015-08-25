########################################################
# commons.py: common functions for all compressive monitors
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/25
########################################################

import numpy as np 
import random


#======================================================#
# Function to remove the entries of data matrix
#======================================================#
def removeEntries(matrix, density, seedID):
    (vecX, vecY) = np.where(matrix > 0)
    vecXY = np.c_[vecX, vecY]
    numRecords = vecX.size
    numAll = matrix.size
    random.seed(seedID)
    randomSequence = range(0, numRecords)
    random.shuffle(randomSequence) # one random sequence per round
    numTrain = int( numAll * density)
    trainXY = vecXY[randomSequence[0 : numTrain], :]
    removedMatrix = np.zeros(matrix.shape)
    removedMatrix[trainXY[:, 0], trainXY[:, 1]] = matrix[trainXY[:, 0], trainXY[:, 1]]
    return removedMatrix
