#########################################################
# run_googledata_cpu.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/24
# Last updated: 2015/8/24
# Implemented approach: JGD: [Silvestri et al., ICDCS'2015]
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
        'dataSample': 'day-sample', # choose 'day-sample', 'week-sample', or 'all'      
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE', 'SNR'], # delete where appropriate   
        'samplingRate': np.arange(0.05, 0.06, 0.05), # sampling rate
        'monitorSelection': 'topW-Update', # monitor selection algorithm
                             # select from 'random', 'topW', 'topW-Update', 'batch-selection'
        'trainingPeriod': 12, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': False, # whether to save log into file
        'debugMode': False, # whether to record the debug info
        'parallelMode': True # whether to leverage multiprocessing for speedup
        }

startTime = time.time() # start timing
utils.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('JGD: [Silvestri et al., ICDCS\'2015].')

# load the dataset
dataMatrix = dataloader.load(para)
dataMatrix = dataMatrix[:,0:24]
# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + utils.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')

