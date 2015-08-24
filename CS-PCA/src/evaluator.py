########################################################
# evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/17
########################################################

import numpy as np 
import time, sys
from util import logger
from commons import evaluatorlib
import cPickle as pickle
import CS_PCA
from commons import resulthandler
import multiprocessing
from matplotlib.pyplot import plot, show, figure, title

#======================================================#
# Function to evalute the approach for xx rounds at each sampling rate
#======================================================#
def execute(matrix, para):
    # loop over each sampling rate and each round
    if para['parallelMode']: # run on multiple processes
        pool = multiprocessing.Pool()
        for rate in para['samplingRate']: 
            for roundId in xrange(para['rounds']):
                pool.apply_async(monitoring, (matrix, rate, roundId, para))
        pool.close()
        pool.join()
    else: # run on single processes
        for rate in para['samplingRate']:
            for roundId in xrange(para['rounds']):
                monitoring(matrix, rate, roundId, para)
    # process the dumped results
    resulthandler.process(para)


#======================================================#
# Function to run compressive monitoring at each density
#======================================================#
def monitoring(matrix, rate, roundId, para):
    startTime = time.clock()
    logger.info('rate=%.2f starts.'%rate)

    # generate trainMatrix, observedMatrix, testMatrix
    (numNode, numTime) = matrix.shape
    trainingPeriod = para['trainingPeriod']
    trainMatrix = matrix[:, 0:trainingPeriod]
    testMatrix = matrix[:, trainingPeriod:] 
    seedID = roundId
    observedMatrix = evaluatorlib.removeEntries(testMatrix, rate, seedID)

    # CS algorithm
    logger.info('CS-PCA estimation...')
    startTime = time.clock() # to record the running time for one round
    recoveredMatrix = CS_PCA.recover(trainMatrix, observedMatrix, para)
    runningTime = float(time.clock() - startTime) 
    
    # calculate the estimation error  
    (testVecX, testVecY) = np.where(testMatrix > 0)
    testVec = testMatrix[testVecX, testVecY]
    estiVec = recoveredMatrix[testVecX, testVecY]
    evalResult = evaluatorlib.errMetric(testVec, estiVec, para['metrics'])
    result = (evalResult, runningTime)
    
    # dump the result at each rate
    outFile = '%s%s%s_result_%.2f%s.tmp'%(para['outPath'], para['dataName'], 
        ('_%s'%para['dataType'] if ('dataType' in para.keys()) else ''), rate, 
        '_round%2d'%(roundId + 1) if (para['rounds'] > 1) else '')
    with open(outFile, 'wb') as fid:
            pickle.dump(result, fid)
    logger.info('rate=%.2f done.'%rate)
    logger.info('----------------------------------------------')




