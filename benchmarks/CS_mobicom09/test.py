from matplotlib.pyplot import plot, show, figure, title
import matplotlib as plt
import numpy as np
from numpy import linalg as LA
from scipy.fftpack import dct, idct, fft, ifft
from scipy.sparse import coo_matrix
from sklearn.linear_model import Lasso 
import sys, random
Fs = 40e3  #Sample rate
duration = 1./8
N_samps = np.floor(duration*Fs)
M = 250   # Number of compressed "basis" functions - we're going from N_samps to M samps.
f1 = 200
f2 = 3950

print "Compression ratio {0}".format(M/N_samps)

t = np.linspace(0,duration,N_samps)

X = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

q = range(0,int(N_samps))
random.shuffle(q);
yi = q[0:M];
yi = np.sort(yi)
#Y = X[yi]
D = fft(np.eye(N_samps))
D_inv = LA.inv(D)
y = np.dot(D, X)
A = D_inv[yi, :]  
Y = np.dot(A, y)
Y = X[yi]
lasso = Lasso(alpha=0.01)
lasso.fit(A,Y)

# plot(lasso.coef_)

print lasso.coef_.shape
Xhat = np.dot(D_inv, lasso.coef_).real
print Xhat.shape
figure(figsize=[12,6])
plot(t,X)
show()
plot(t,Xhat)

show()










