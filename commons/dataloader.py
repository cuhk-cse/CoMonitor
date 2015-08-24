########################################################
# dataloader.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/17
########################################################

import numpy as np 
import os, sys
from util import logger


#======================================================#
# Function to load the dataset
#======================================================#
def load(para):
    if para['dataName'] == 'google-cluster-data':
        datafile = para['dataPath'] + para['dataName'] + '/data-analyzer/machine-usage/'\
            + para['dataSample'] + '/machine_%s_usage_matrix.csv'%para['dataType']
        logger.info('Loading data: %s'%os.path.abspath(datafile))
        data = np.loadtxt(datafile, delimiter = ',')
        dataMatrix = preprocess(data, para)
    elif para['dataName'] == 'synthetic_data_icdcs15':
        datafile = para['dataPath'] + para['dataName'] + '.txt'
        logger.info('Loading data: %s'%os.path.abspath(datafile))
        data = np.loadtxt(datafile)
        dataMatrix = preprocess(data, para)
    elif para['dataName'] == 'ndbc-ctd':
        datafile = para['dataPath'] + para['dataName'] + '/CTD_7.0N180W_0803290541.cor'
        logger.info('Loading data: %s'%os.path.abspath(datafile))
        data = np.genfromtxt(datafile, dtype=np.float64, skip_header=38)
        dataMatrix = data[:, 1:2] # extract temperature data
    elif para['dataName'] == 'intellab_data':
        datafile = para['dataPath'] + para['dataName'] + '.txt'
        logger.info('Loading data: %s'%os.path.abspath(datafile))
        data = np.genfromtxt(datafile, dtype=np.float64, delimiter=' ', missing_values='', 
            usecols=(2,3,4), skip_footer=526)
        dataMatrix = preprocess(data, para)
    elif para['dataName'] == 'Orangelab_sense_temperature':
        datafile = para['dataPath'] + para['dataName'] + '.txt'
        data = np.loadtxt(datafile)
        print data.shape
        dataMatrix = preprocess(data, para)
    else:
        logger.error('Data file not found!')
        sys.exit() 
    logger.info('Data size: %d nodes * %d timeslices'\
        %(dataMatrix.shape[0], dataMatrix.shape[1]))
    logger.info('Loading data done.')
    logger.info('----------------------------------------------') 
    return dataMatrix


#======================================================#
# Function to preprocess the dataset
# delete the invalid values
#======================================================#
def preprocess(matrix, para):
    if para['dataName'] == 'google-cluster-data':
        matrix = np.where(matrix < 0, 0, matrix)
        matrix = np.where(matrix > 1, 1, matrix)
        idx = np.sum(matrix <= 0, axis=1) == 0
        dataMatrix = matrix[idx, :]
    elif para['dataName'] == 'synthetic_data_icdcs15':
        matrix = matrix[:, 2:]
        dataMatrix = matrix.T
    elif para['dataName'] == 'intellab_data':
        dataMatrix = np.zeros((54, 65535))
        for i in xrange(matrix.shape[0]):
            if matrix[i, 1] <= 54 and matrix[i, 2] < 50 and matrix[i, 2] > 0:
                dataMatrix[int(matrix[i, 1]) - 1, int(matrix[i, 0]) - 1] = matrix[i, 2] 
        idx = np.sum(dataMatrix <= 0, axis=0) <= 10
        dataMatrix = dataMatrix[:, idx]
    elif para['dataName'] == 'Orangelab_sense_temperature':
        idx = np.sum(matrix <= 0, axis=1) == 0
        dataMatrix = matrix[idx, :]
    else:
        dataMatrix = matrix
    return dataMatrix

