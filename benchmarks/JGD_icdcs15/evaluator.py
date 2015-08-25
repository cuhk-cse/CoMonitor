########################################################
# evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/17
########################################################

import numpy as np 
from numpy import linalg as LA
import time
from commons.utils import logger
from commons import evallib
import cPickle as pickle
from comonitor import JGD
import multiprocessing


#======================================================#
# Function to evalute the approach for xx rounds at each 
# sampling rate
#======================================================#
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
    # summarize the dumped results
    evallib.summarizeResult(para)
#======================================================#


#======================================================#
# Function to run the compressive monitoring at each density
#======================================================#
def monitoring(matrix, rate, para):
    startTime = time.clock()
    logger.info('rate=%.2f starts.'%rate) 

    # JGD algorithm
    startTime = time.clock() # to record the running time for one round
    trainingPeriod = para['trainingPeriod']
    trainMatrix = matrix[:, 0:trainingPeriod]
    testMatrix = matrix[:, trainingPeriod:]
    logger.info('monitor selection...')
    (selectedMonitors, toEstimateNodes) = JGD.selectMonitor(trainMatrix, rate, para)
    observedMatrix = testMatrix[selectedMonitors, :] 
    logger.info('JGD estimation...')
    recoveredMatrix = JGD.recover(trainMatrix, observedMatrix, selectedMonitors, toEstimateNodes)
    runningTime = float(time.clock() - startTime) 
    
    # evaluate the estimation error  
    evalResult = evallib.evaluate(testMatrix, recoveredMatrix, para)
    result = (evalResult, runningTime)
    
    # dump the result at each rate
    outFile = '%s%s%s_result_%.2f.tmp'%(para['outPath'], para['dataName'], 
        '_%s'%para['dataType'] if ('dataType' in para.keys()) else '', rate)
    evallib.dumpresult(outFile, result)
    
    logger.info('rate=%.2f done.'%rate)
    logger.info('----------------------------------------------') 
