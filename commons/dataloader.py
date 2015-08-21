########################################################
# dataloader.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/17
########################################################

import numpy as np 
import sys
from util import * # import logger


########################################################
# Function to load the dataset
#
def load(para):
    if para['dataName'] == 'google-cluster-data':
        datafile = para['dataPath'] + para['dataName'] + '/data-analyzer/machine-usage/'\
            + para['dataSample'] + '/machine_%s_usage_matrix_sample.csv'%para['dataType']
        logger.info('Loading data: %s'%os.path.abspath(datafile))
        dataMatrix = np.loadtxt(datafile, delimiter = ',')
    elif para['dataName'] == 'synthetic_data_icdcs15':
        datafile = para['dataPath'] + para['dataName'] + '.txt'
        logger.info('Loading data: %s'%os.path.abspath(datafile))
        dataMatrix = np.loadtxt(datafile)
    elif para['dataName'] == 'ndbc-ctd':
        datafile = para['dataPath'] + para['dataName'] + '/CTD_7.0N180W_0803290541.cor'
        logger.info('Loading data: %s'%os.path.abspath(datafile))
        data = np.genfromtxt(datafile, dtype=np.float64, skip_header=38)
        dataMatrix = data[:, 1:2] # temperature data
    else:
        logger.error('Data file not found!')
        sys.exit()
    dataMatrix = preprocess(dataMatrix, para)
    logger.info('Loading data done.')
    logger.info('Data size: %d nodes * %d timeslices'\
        %(dataMatrix.shape[0], dataMatrix.shape[1]))
    return dataMatrix
########################################################


########################################################
# Function to preprocess the dataset
# delete the invalid values
# 
def preprocess(matrix, para):
    if para['dataName'] == 'google-cluster-data':
        matrix = np.where(matrix < 0, 0, matrix)
        matrix = np.where(matrix > 1, 1, matrix)
        idx = (np.sum(matrix > 0, axis=1) / float(matrix.shape[1]) > 0.05)
        matrix = matrix[idx, :]
    elif para['dataName'] == 'synthetic_data_icdcs15':
        matrix = matrix[:, 2:]
        matrix = matrix.T
    return matrix
########################################################

