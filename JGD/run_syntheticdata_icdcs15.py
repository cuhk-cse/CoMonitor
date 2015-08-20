#########################################################
# run_rt.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/7/16
# Implemented approaches: ADF [Wu et al., TSMC'2013]
# Evaluation metrics: MAE, NMAE, RMSE, MRE, NPRE
#########################################################


import numpy as np
import os, sys, time
sys.path.append('../commons/')
from util import * 
import dataloader
from src import evaluator

#########################################################
# config area
#
para = {'dataPath': '../data/', # data path
        'dataName': 'synthetic_data_icdcs15', # set the dataset name    
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE'], # delete where appropriate   
        'samplingRate': list(np.arange(0.05, 0.96, 0.05)), # sampling rate
        'monitorSelection': 'batch-selection', # monitor selection algorithm
                               # select from 'random', 'topW', 'topW-Update', 'batch-selection'
        'trainingPeriod': 500, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': True, # whether to save log into file
        'debugMode': False, #whether to record the debug info
        'parallelMode': False # whether to leverage multiprocessing for speedup
        }

initConfig(para)
#########################################################


startTime = time.time() # start timing
logger.info('==============================================')
logger.info('JGD: [Silvestri et al., ICDCS\'2015].')

# load the dataset
dataMatrix = dataloader.load(para)

# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')
sys.path.remove('../commons/')

