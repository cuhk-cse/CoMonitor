#########################################################
# run_ctddata_mobicom09.py 
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/24
# Last updated: 2015/8/24
# Implemented approaches: CS [Luo et al., MobiCom'2009]
#########################################################


import numpy as np
import os, sys, time
sys.path.append('../')
from commons.utils import logger
from commons import utils
from commons import dataloader
import evaluator
import pywt
from numpy import linalg as LA
from matplotlib.pyplot import plot, show, figure, title

# parameter config area
para = {'dataPath': '../data/', # data path
        'dataName': 'ndbc-ctd', # set the dataset name     
        'outPath': 'result/', # output path for results
        'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NNPRE', 'SNR'], # delete where appropriate   
        'samplingRate': np.arange(0.05, 0.96, 0.05), # sampling rate
        'rounds': 1, # how many runs to perform at each sampling rate
        'transform': 'DCT', # transform base: 'DCT' or 'DWT'
        'lmbda': 1e-6, # sparisty regularization parameter
        'trainingPeriod': 0, # training time periods
        'saveTimeInfo': False, # whether to keep track of the running time
        'saveLog': False, # whether to save log into file
        'debugMode': False, #whether to record the debug info
        'parallelMode': False # whether to leverage multiprocessing for speedup
        }

startTime = time.time() # start timing
util.setConfig(para) # set configuration
logger.info('==============================================')
logger.info('Compressive Sensing: [Luo et al., MobiCom\'2009].')

# load the dataset
dataMatrix = dataloader.load(para)
# dataMatrix[:,0] = sorted(dataMatrix[:,0], reverse=True)
# x = np.array([1, 2, 3, 4, 5, 6])
# y = pywt.dwt2([1,2], 'haar')
# print y
# # y = [x for x in y] 
# # plot(y, 'o')
# show()
# sys.exit()
# evaluate compressive monitoring algorithm
evaluator.execute(dataMatrix, para)

logger.info('All done. Elaspsed time: ' + util.formatElapsedTime(time.time() - startTime)) # end timing
logger.info('==============================================')

