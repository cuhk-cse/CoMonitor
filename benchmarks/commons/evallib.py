########################################################
# evallib.py: common functions for evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/21
########################################################

import numpy as np 
from numpy import linalg as LA
import os, sys, time
import cPickle as pickle
from commons.utils import logger


#======================================================#
# Function to compute the evaluation metrics
#======================================================#
def evaluate(testMatrix, recoveredMatrix, para):
    (testVecX, testVecY) = np.where(testMatrix > 0)
    testVec = testMatrix[testVecX, testVecY]
    estiVec = recoveredMatrix[testVecX, testVecY]
    evalResult = errMetric(testVec, estiVec, para['metrics'])
    return evalResult


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
                npre = relativeError[int(np.floor(0.99 * relativeError.shape[0]))] 
                result = np.append(result, npre)
        if 'SNR' == metric:
            snr = 10 * np.log10(np.sum(realVec **2) / np.sum((realVec - estiVec) **2))
            result = np.append(result, snr)
    return result


#======================================================#
# Dump the raw result into tmp file
#======================================================#
def dumpresult(outFile, result):
    try:
        with open(outFile, 'wb') as fid:
                pickle.dump(result, fid)
    except Exception, e:
        logger.error('Dump file failed: ' + outFile)
        logger.error(e)
        sys.exit()


#======================================================#
# Process the raw result files 
#======================================================#
def summarizeResult(para):
    if 'rounds' not in para.keys():
        para['rounds'] = 1
    path = '%s%s%s_result'%(para['outPath'], para['dataName'], 
        '_%s'%para['dataType'] if ('dataType' in para.keys()) else '')
    evalResults = np.zeros((len(para['samplingRate']), para['rounds'], len(para['metrics']))) 
    timeResults = np.zeros((len(para['samplingRate']), para['rounds']))   
    k = 0
    
    print '===== Average result summary ====='
    print 'Metrics:', para['metrics'] 

    for rate in para['samplingRate']:
        for rnd in xrange(para['rounds']):
            inputfile = path + '_%.2f%s.tmp'%(rate, 
                '_round%02d'%(rnd + 1) if (para['rounds'] > 1) else '')
            with open(inputfile, 'rb') as fid:
                data = pickle.load(fid)
            os.remove(inputfile)
            (evalResults[k, rnd, :], timeResults[k, rnd]) = data
        print 'rate=%.2f: '%rate, np.average(evalResults[k, :, :], axis=0)
        k += 1
    saveSummaryResult(path, evalResults, timeResults, para)  


#======================================================#
# Save the summary evaluation results into file
#======================================================#
def saveSummaryResult(outfile, result, timeinfo, para):
    fileID = open(outfile + '.txt', 'w')
    fileID.write('======== Average result summary ========\n')
    fileID.write('Metric:  ')
    for metric in para['metrics']:
        fileID.write('|  %s  '%metric)
    fileID.write('\n')
    k = 0
    for rate in para['samplingRate']:
        fileID.write('rate=%.2f: '%rate)
        np.savetxt(fileID, np.matrix(np.average(result[k, :, :], axis=0)), fmt='%.4f', delimiter='  ')
        k += 1

    if result.shape[1] > 1: # i.e. para['rounds'] > 1
        fileID.write('\n======== Detailed results ========\n')
        k = 0
        for rate in para['samplingRate']:
            fileID.write('rate=%.2f: %2d rounds\n'%(rate, para['rounds']))
            fileID.write('------------------------------------------------\n')
            np.savetxt(fileID, np.matrix(result[k, :, :]), fmt='%.4f', delimiter='  ')
            fileID.write('\n')
            k += 1
    fileID.close()

    if para['saveTimeInfo']:
        fileID = open(outfile + '_time.txt', 'w')
        fileID.write('Average running time (second):\n')
        k = 0
        for rate in para['samplingRate']:
            fileID.write('rate=%.2f: '%rate)
            np.savetxt(fileID, np.matrix(np.average(timeinfo[k, :])), fmt='%.4f', delimiter='  ')  
            k += 1
        fileID.close()
