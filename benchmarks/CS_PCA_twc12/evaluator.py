########################################################
# evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/17
########################################################

import numpy as np 
import time, sys
from commons.utils import logger
from commons import evallib
import cPickle as pickle
from comonitor import CS_PCA
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
            for roundId in xrange(para['rounds']):
                pool.apply_async(monitoring, (matrix, rate, roundId, para))
        pool.close()
        pool.join()
    else: # run on single processes
        for rate in para['samplingRate']:
            for roundId in xrange(para['rounds']):
                monitoring(matrix, rate, roundId, para)
    # summarize the dumped results
    evallib.summarizeResult(para)


#======================================================#
# Function to run compressive monitoring at each density
#======================================================#
def monitoring(matrix, rate, roundId, para):
    startTime = time.clock()
    logger.info('rate=%.2f starts.'%rate)

    # sampling
    logger.info('CS-PCA sampling...')
    startTime = time.clock() # to record the running time for one round
    (trainMatrix, observedMatrix, testMatrix) = CS_PCA.sampling(matrix, rate, roundId, para)

    # CS-PCA algorithm
    logger.info('CS-PCA estimation...')
    recoveredMatrix = CS_PCA.recover(trainMatrix, observedMatrix, para)
    runningTime = float(time.clock() - startTime) 
    
    # evaluate the estimation error  
    evalResult = evallib.evaluate(testMatrix, recoveredMatrix, para)
    result = (evalResult, runningTime)
    
    # dump the result at each rate
    outFile = '%s%s%s_result_%.2f%s.tmp'%(para['outPath'], para['dataName'], 
        ('_%s'%para['dataType'] if ('dataType' in para.keys()) else ''), rate, 
        '_round%2d'%(roundId + 1) if (para['rounds'] > 1) else '')
    evallib.dumpresult(outFile, result)
    
    logger.info('rate=%.2f done.'%rate)
    logger.info('----------------------------------------------')




