########################################################
# CS_PCA.py: PCA based CS estimation
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2015/8/17
# Last updated: 2015/8/23
########################################################

import numpy as np 
from numpy import linalg as LA
import time
import random
from sklearn.linear_model import Lasso 


#======================================================#
# Function to recover the unobserved values
#======================================================#
def recover(trainMatrix, observedMatrix, para):
    (numNode, numTestTime) = observedMatrix.shape
    avgVec = np.average(trainMatrix, axis=1)
    covMatrix = np.cov(trainMatrix)
    (U, s, V) = LA.svd(covMatrix)
    transformMatrix = U
    recoveredMatrix = np.zeros((numNode, numTestTime))
    for i in xrange(numTestTime):
        # transform
        (monitors, ) = np.nonzero(observedMatrix[:, i] > 0)
        observedVec = observedMatrix[monitors, i]
        measureMatrix = transformMatrix[monitors, :]
        # lasso recovery
        lasso = Lasso(alpha=para['lmbda'])
        lasso.fit(measureMatrix, observedVec - avgVec[monitors])
        recoveredMatrix[:, i] = np.dot(transformMatrix, lasso.coef_) + avgVec
    recoveredMatrix[observedMatrix > 0] = observedMatrix[observedMatrix > 0]
    return recoveredMatrix
