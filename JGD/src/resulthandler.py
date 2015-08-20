########################################################
# resulthandler.py: get the average values of the results
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2015/8/18
########################################################

import numpy as np
import linecache
import os, sys, time
import cPickle as pickle
 

########################################################
# Process the raw results 
#
def process(para):
    if 'dataType' in para.keys():
        path = '%s%s_%s_result'%(para['outPath'], para['dataName'], para['dataType'])
    else: 
        path = '%s%s_result'%(para['outPath'], para['dataName'])
    evalResults = np.zeros((len(para['samplingRate']), len(para['metrics']))) 
    timeResults = np.zeros((len(para['samplingRate']), 1))   
    k = 0
    
    print '===== Average result summary ====='
    print 'Metrics:', para['metrics'] 

    for rate in para['samplingRate']:
        inputfile = path + '_%.2f.tmp'%rate
        with open(inputfile, 'rb') as fid:
            data = pickle.load(fid)
        os.remove(inputfile)
        (evalResults[k, :], timeResults[k]) = data
        k += 1
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        print 'rate=%.2f: '%rate, data[0]

    saveResult(path, evalResults, timeResults, para)  
########################################################


########################################################
# Save the evaluation results into file
#
def saveResult(outfile, result, timeinfo, para):
    fileID = open(outfile + '.txt', 'w')
    fileID.write('Metric:  ')
    for metric in para['metrics']:
        fileID.write('|  %s  '%metric)
    fileID.write('\n')
    k = 0
    for rate in para['samplingRate']:
        fileID.write('rate=%.2f: '%rate)
        np.savetxt(fileID, np.matrix(result[k, :]), fmt='%.4f', delimiter='  ')
        k += 1 
    fileID.close()

    if para['saveTimeInfo']:
        fileID = open(outfile + '_time.txt', 'w')
        fileID.write('Average running time (second):\n')
        k = 0
        for rate in para['samplingRate']:
            fileID.write('rate=%.2f: '%rate)
            np.savetxt(fileID, np.matrix(timeinfo[k]), fmt='%.4f', delimiter='  ')  
        fileID.close()
########################################################
