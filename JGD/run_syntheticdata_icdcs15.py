#########################################################
# run_syntheticdata_icdcs15.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/24
# Last updated: 2015/8/24
# Implemented approach: JGD [Silvestri et al., ICDCS'2015]
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
        'dataName': 'synthetic_data_icdcs15', # set the dataset name    
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE', 'SNR'], # delete where appropriate   
        'samplingRate': np.arange(0.05, 0.96, 0.05), # sampling rate
        'monitorSelection': 'topW-Update', # monitor selection algorithm
                           # select from 'random', 'topW', 'topW-Update', 'batch-selection'
        'trainingPeriod': 400, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': False, # whether to save log into file
        'debugMode': False, #whether to record the debug info
        'parallelMode': False # whether to leverage multiprocessing for speedup
        }

startTime = time.time() # start timing
util.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('JGD: [Silvestri et al., ICDCS\'2015].')

# load the dataset
dataMatrix = dataloader.load(para)

# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + util.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')

