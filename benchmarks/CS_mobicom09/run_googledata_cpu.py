#########################################################
# run_googledata_cpu.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/24
# Last updated: 2015/8/24
#########################################################


import numpy as np
import os, sys, time
sys.path.append('../')
from commons.utils import logger
from commons import utils
from commons import dataloader
import evaluator


# parameter config area
para = {'dataPath': '../data/', # data path
        'dataName': 'google-cluster-data', # set the dataset name
        'dataType': 'cpu', # data type: cpu or memory
        'dataSample': 'day-sample', # choose 'day-sample', 'week-sample', or 'all-data'   
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE', 'SNR'], # evaluation metrics  
        'samplingRate': np.arange(0.1, 0.11, 0.05), # sampling rate
        'rounds': 1, # how many runs to perform at each sampling rate
        'transform': 'DCT', # transform base: 'DCT', 'DWT-haar', or 'DWT-db4'
        'lmbda': 1e-4, # sparisty regularization parameter
        'trainingPeriod': 12, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': False, # whether to save log into file
        'debugMode': False, #whether to record the debug info
        'parallelMode': True # whether to leverage multiprocessing for speedup
        }

startTime = time.time() # start timing
utils.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('Compressive Sensing: [Luo et al., MobiCom\'2009].')

# load the dataset
dataMatrix = dataloader.load(para)
dataMatrix = dataMatrix[:,0:24]
# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + utils.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')

