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
def removeEntries(matrix, density, roundId):
    (numRow, numCol) = matrix.shape
    seedId = roundId
    removedMatrix = np.zeros(matrix.shape)
    for i in xrange(numCol):
        randomSequence = range(0, numRow)
        random.seed(seedId + i)
        random.shuffle(randomSequence) # one random sequence 
        numSample = int(numRow * density)
        sampleIdx = randomSequence[0 : numSample]
        removedMatrix[sampleIdx, i] = matrix[sampleIdx, i] 
    return removedMatrix
