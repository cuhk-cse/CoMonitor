########################################################
# resulthandler.py: get the average values of the results
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2015/8/18
########################################################

import numpy as np
import os, sys, time
import cPickle as pickle
 

#======================================================#
# Process the raw results 
#======================================================#
def process(para):
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
                '_round%2d'%(rnd + 1) if (para['rounds'] > 1) else '')
            with open(inputfile, 'rb') as fid:
                data = pickle.load(fid)
            os.remove(inputfile)
            (evalResults[k, rnd, :], timeResults[k, rnd]) = data
        print 'rate=%.2f: '%rate, np.average(evalResults[k, :, :], axis=0)
        k += 1

    saveResult(path, evalResults, timeResults, para)  


#======================================================#
# Save the evaluation results into file
#======================================================#
def saveResult(outfile, result, timeinfo, para):
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
