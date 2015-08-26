#########################################################
# run_orangelab_temperature.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/24
# Last updated: 2015/8/24
# Implemented approach: CS-PCA [Quer et al., TWC'2012]
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
        'dataName': 'Orangelab_sense_temperature', # set the dataset name     
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE', 'SNR'], # evaluation metrics  
        'samplingRate': np.arange(0.05, 0.96, 0.05), # sampling rate
        'rounds': 1, # how many runs to perform at each sampling rate
        'lmbda': 1e-5, # sparisty regularization parameter
        'trainingPeriod': 33, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': False, # whether to save log into file
        'debugMode': False, #whether to record the debug info
        'parallelMode': False # whether to leverage multiprocessing for speedup
        }

startTime = time.time() # start timing
utils.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('CS-PCA: [Quer et al., TWC\'2012]')

# load the dataset
dataMatrix = dataloader.load(para)

# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + utils.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')

