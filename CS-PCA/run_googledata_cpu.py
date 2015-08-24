#########################################################
# run_googledata_cpu.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/24
# Last updated: 2015/8/24
# Implemented approach: CS-PCA [Quer et al., TWC'2012]
#########################################################


import numpy as np
import os, sys, time
sys.path.append('../')
from commons.util import logger
from commons import util
from commons import dataloader
from src import evaluator


# parameter config area
para = {'dataPath': '../data/', # data path
        'dataName': 'google-cluster-data', # set the dataset name
        'dataType': 'cpu', # data type: cpu or memory
        'dataSample': 'day-sample', # choose 'day-sample', 'week-sample', or 'all-data'   
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE', 'SNR'], # evaluation metrics  
        'samplingRate': np.arange(0.05, 0.95, 0.05), # sampling rate
        'rounds': 20, # how many runs to perform at each sampling rate
        'lmbda': 1e-4, # sparisty regularization parameter
        'trainingPeriod': 144, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': False, # whether to save log into file
        'debugMode': False, #whether to record the debug info
        'parallelMode': False # whether to leverage multiprocessing for speedup
        }

startTime = time.time() # start timing
util.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('CS-PCA: [Quer et al., TWC\'2012]')

# load the dataset
dataMatrix = dataloader.load(para)

# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + util.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')

