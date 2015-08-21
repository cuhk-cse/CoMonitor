########################################################
# evaluatorlib.py: common functions for evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/21
########################################################

import numpy as np 
import random
from numpy import linalg as LA

########################################################
# Function to remove the entries of data matrix
# Return the trainMatrix and the corresponding testing data
#
def removeEntries(matrix, density, seedID):
    (vecX, vecY) = np.where(matrix > 0)
    vecXY = np.c_[vecX, vecY]
    numRecords = vecX.size
    numAll = matrix.size
    random.seed(seedID)
    randomSequence = range(0, numRecords)
    random.shuffle(randomSequence) # one random sequence per round
    numTrain = int( numAll * density)
    # by default, we set the remaining QoS records as testing data                     
    numTest = numRecords - numTrain
    trainXY = vecXY[randomSequence[0 : numTrain], :]
    testXY = vecXY[randomSequence[- numTest :], :]

    trainMatrix = np.zeros(matrix.shape)
    trainMatrix[trainXY[:, 0], trainXY[:, 1]] = matrix[trainXY[:, 0], trainXY[:, 1]]
    testMatrix = np.zeros(matrix.shape)
    testMatrix[testXY[:, 0], testXY[:, 1]] = matrix[testXY[:, 0], testXY[:, 1]]

    # ignore invalid testing data: handling all empty rows and columns
    if trainMatrix.shape[1] > 1:
        idxX = (np.sum(trainMatrix, axis=1) == 0)
        testMatrix[idxX, :] = 0
        idxY = (np.sum(trainMatrix, axis=0) == 0)
        testMatrix[:, idxY] = 0
    return trainMatrix, testMatrix
########################################################


########################################################
# Function to compute the evaluation metrics
#
def errMetric(realVec, estiVec, metrics):
    result = []
    absError = np.abs(estiVec - realVec) 
    mae = np.sum(absError)/absError.shape
    for metric in metrics:
        if 'MAE' == metric:
            result = np.append(result, mae)
        if 'NMAE' == metric:
            nmae = mae / (np.sum(realVec) / absError.shape)
            result = np.append(result, nmae)
        if 'RMSE' == metric:
            rmse = LA.norm(absError) / np.sqrt(absError.shape)
            result = np.append(result, rmse)
        if 'MRE' == metric or 'NNPRE' == metric:
            relativeError = absError / realVec
            if 'MRE' == metric:
                mre = np.average(relativeError)
                result = np.append(result, mre)
            if 'NNPRE' == metric:
                relativeError = np.sort(relativeError)
                npre = relativeError[np.floor(0.99 * relativeError.shape[0])] 
                result = np.append(result, npre)
    return result
########################################################

