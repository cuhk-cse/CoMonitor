#########################################################
# run_orangelab_temperature.py 
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
        'dataName': 'EPFL_LUCE_temperature', # set the dataset name     
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE', 'SNR'], # evaluation metrics  
        'samplingRate': np.arange(0.05, 0.96, 0.05), # sampling rate
        'monitorSelection': 'topW-Update', # monitor selection algorithm
                           # select from 'random', 'topW', 'topW-Update', 'batch-selection'
        'trainingPeriod': 40, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': False, # whether to save log into file
        'debugMode': False, #whether to record the debug info
        'parallelMode': False # whether to leverage multiprocessing for speedup
        }

startTime = time.time() # start timing
utils.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('Baseline approach.')

# load the dataset
dataMatrix = dataloader.load(para)

# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + utils.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')

