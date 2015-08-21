########################################################
# evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/17
########################################################

import numpy as np 
from numpy import linalg as LA
import time
from util import * # import logger
import evaluatorlib
import cPickle as pickle
import JGD
import resulthandler
import multiprocessing


########################################################
# Function to evalute the approach for xx rounds at each sampling rate
# 
def execute(matrix, para):
    # loop over each sampling rate and each round
    if para['parallelMode']: # run on multiple processes
        pool = multiprocessing.Pool()
        for rate in para['samplingRate']: 
            pool.apply_async(monitoring, (matrix, rate, para))
        pool.close()
        pool.join()
    else: # run on single processes
        for rate in para['samplingRate']:
            monitoring(matrix, rate, para)
    # process the dumped results
    resulthandler.process(para)
########################################################


########################################################
# Function to run the compressive monitoring at each density
# 
def monitoring(matrix, rate, para):
    startTime = time.clock()
    logger.info('rate=%.2f starts.'%rate)
    logger.info('----------------------------------------------') 
    (numNode, numTime) = matrix.shape
    trainingPeriod = para['trainingPeriod']
    trainMatrix = matrix[:, 0:trainingPeriod]
    testMatrix = matrix[:, trainingPeriod:]

    # JGD algorithm
    startTime = time.clock() # to record the running time for one round
    (selectedMonitors, toEstimateNodes) = JGD.selectMonitor(trainMatrix, rate, para)
    logger.info('monitor selection done.')
    observedMatrix = testMatrix[selectedMonitors, :]
    recoveredMatrix = JGD.recover(trainMatrix, observedMatrix, selectedMonitors, toEstimateNodes)
    logger.info('JGD estimation done.')
    runningTime = float(time.clock() - startTime) 
    
    # calculate the estimation error  
    (testVecX, testVecY) = np.where(testMatrix > 0)
    testVec = testMatrix[testVecX, testVecY]
    estiVec = recoveredMatrix[testVecX, testVecY]
    evalResult = evaluatorlib.errMetric(testVec, estiVec, para['metrics'])
    result = (evalResult, runningTime)
    
    # dump the result at each rate
    if 'dataType' in para.keys():
        outFile = '%s%s_%s_result_%.2f.tmp'%(para['outPath'], para['dataName'], para['dataType'], rate)
    else: 
        outFile = '%s%s_result_%.2f.tmp'%(para['outPath'], para['dataName'], rate)
    with open(outFile, 'wb') as fid:
            pickle.dump(result, fid)
    logger.info('rate=%.2f done.'%rate)
    logger.info('----------------------------------------------') 
########################################################
