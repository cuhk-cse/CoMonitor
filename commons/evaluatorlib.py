########################################################
# evaluatorlib.py: common functions for evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/21
########################################################

import numpy as np 
import random
from numpy import linalg as LA


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
    trainMatrix = np.zeros(matrix.shape)
    trainMatrix[trainXY[:, 0], trainXY[:, 1]] = matrix[trainXY[:, 0], trainXY[:, 1]]
    return trainMatrix


#======================================================#
# Function to compute the evaluation metrics
#======================================================#
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
        if 'SNR' == metric:
            snr = 10 * np.log10(np.sum(realVec **2) / np.sum((realVec - estiVec) **2))
            result = np.append(result, snr)
    return result

