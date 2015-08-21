########################################################
# JGD.py: joint Guassian distribution based estimation
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/17
########################################################

import numpy as np 
from numpy import linalg as LA
import time
import random
from util import *


########################################################
# Function to select optimal node monitors
# 
def selectMonitor(trainMatrix, rate, para):
    numNodes = trainMatrix.shape[0]
    numMonitors = int(round(rate * numNodes))
    if para['monitorSelection'] == 'random':
        random.seed(1) # fix seed at each rate
        randomperm = range(0, numNodes)
        random.shuffle(randomperm) 
        selectedMonitors = randomperm[0:numMonitors]
        toEstimateNodes = randomperm[numMonitors:]
    elif para['monitorSelection'] == 'batch-selection':
        (selectedMonitors, toEstimateNodes) = batchSelection(trainMatrix, rate, para)
    elif para['monitorSelection'] == 'topW':
        (selectedMonitors, toEstimateNodes) = topW(trainMatrix, rate, para)
    elif para['monitorSelection'] == 'topW-Update':
        (selectedMonitors, toEstimateNodes) = topW_Update(trainMatrix, rate, para)
    logger.debug('monitor nodes: ' + str(selectedMonitors))
    logger.debug('remaining nodes: ' + str(toEstimateNodes))
    return selectedMonitors, toEstimateNodes
########################################################


########################################################
# Function to select node monitors with batch selection algorithm
# 
def batchSelection(trainMatrix, rate, para):
    numNodes = trainMatrix.shape[0]
    numMonitors = int(round(rate * numNodes))
    covMatrix = np.cov(trainMatrix, bias=1)
    toEstimateNodes = range(0, numNodes)
    selectedMonitors = []
    for i in xrange(numMonitors):
        minObj = np.inf
        for j in xrange(numNodes - i):
            Yl = list(toEstimateNodes) # copy list
            Sl = list(selectedMonitors) # copy list
            Sl.append(Yl[j])
            Yl.pop(j)
            covMatrix_Yl_Yl = covMatrix[np.ix_(Yl, Yl)]
            covMatrix_Yl_Sl = covMatrix[np.ix_(Yl, Sl)]
            covMatrix_Sl_Sl = covMatrix[np.ix_(Sl, Sl)]
            covMatrix_Sl_Yl = covMatrix[np.ix_(Sl, Yl)]
            obj = np.trace(covMatrix_Yl_Yl\
                - np.dot(np.dot(covMatrix_Yl_Sl, LA.inv(covMatrix_Sl_Sl)), covMatrix_Sl_Yl))
            if obj < minObj:
                xl_star = j
                minObj = obj
        selectedMonitors.append(toEstimateNodes[xl_star])
        toEstimateNodes.pop(xl_star)
    return selectedMonitors, toEstimateNodes
########################################################


########################################################
# Function to select node monitors with topW algorithm
# 
def topW(trainMatrix, rate, para):
    numNodes = trainMatrix.shape[0]
    numMonitors = int(round(rate * numNodes))
    covMatrix = np.cov(trainMatrix, bias=1)
    weightVec = np.sum(covMatrix **2, axis=1) / np.diag(covMatrix)
    idx = np.argsort(-weightVec)
    selectedMonitors = idx[0:numMonitors]
    toEstimateNodes = idx[numMonitors:]
    return selectedMonitors, toEstimateNodes
########################################################


########################################################
# Function to select node monitors with topW-Update algorithm
# 
def topW_Update(trainMatrix, rate, para):
    numNodes = trainMatrix.shape[0]
    numMonitors = int(round(rate * numNodes))
    covMatrix = np.cov(trainMatrix, bias=1)
    toEstimateNodes = range(0, numNodes)
    selectedMonitors = []
    for i in xrange(numMonitors):
        weightVec = np.sum(covMatrix **2, axis=1) / np.diag(covMatrix)
        xl = np.argmax(weightVec)
        Y = range(0, numNodes - i)
        Y.pop(xl)
        xl_star = toEstimateNodes[xl]
        selectedMonitors.append(xl_star)
        toEstimateNodes.remove(xl_star)
        covMatrix_Y_Y = covMatrix[np.ix_(Y, Y)]
        covMatrix_Y_xl = covMatrix[np.ix_(Y, [xl])]
        covMatrix_xl_xl = covMatrix[np.ix_([xl], [xl])]
        covMatrix_xl_Y = covMatrix[np.ix_([xl], Y)]
        covMatrix = covMatrix_Y_Y - np.dot(np.dot(covMatrix_Y_xl, LA.inv(covMatrix_xl_xl)), 
            covMatrix_xl_Y)
    return selectedMonitors, toEstimateNodes
########################################################


########################################################
# Function to recover the unobserved values
# 
def recover(trainMatrix, observedMatrix, selectedMonitors, toEstimateNodes):
    numTestTime = observedMatrix.shape[1]
    avgVec = np.average(trainMatrix, axis=1)
    monitorAvgVec = avgVec[selectedMonitors]
    toEstiAvgVec = avgVec[toEstimateNodes]
    covMatrix = np.cov(trainMatrix, bias=1)
    covMatrix_YS = covMatrix[np.ix_(toEstimateNodes, selectedMonitors)]
    covMatrix_SS = covMatrix[np.ix_(selectedMonitors, selectedMonitors)]
    estiMatrix = np.tile(toEstiAvgVec, (numTestTime, 1)).T\
        + np.dot((np.dot(covMatrix_YS, LA.inv(covMatrix_SS))), 
                observedMatrix - np.tile(monitorAvgVec, (numTestTime, 1)).T)
    recoveredMatrix = np.zeros((trainMatrix.shape[0], numTestTime))
    recoveredMatrix[selectedMonitors, :] = observedMatrix
    recoveredMatrix[toEstimateNodes, :] = estiMatrix
    return recoveredMatrix
########################################################
