import numpy as np
import os
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import misc
import matplotlib.pyplot as plt

def pca(threshold,X):
	#received an nxd2 matrix
	mean = np.mean(X,axis=0)
	cov_matrix = (X-mean).dot((X-mean).T)
	#cov matrix is nxn
	w, v = LA.eig(cov_matrix)
	# n eigenvalues and nxn eigenvectors 
	# v[:,i] is eigenvector corresponding to eigenvalue w[i]
	mean = np.mean(X,axis=0)
	A = np.subtract(X,mean)
	# print A.shape
	eigenvectors = np.dot(A.T,v)
	for i in range(len(X)):
		eigenvectors[:,i]/=np.linalg.norm(eigenvectors[:,i])
	# np.linalg.norm(u[:,i])
	eigenvalues = w
	idx = eigenvalues.argsort()[::-1]  
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	total = np.sum(eigenvalues)
	k_eigienvalues = 0
	k=0
	new_eigenvectors=[]
	while(k_eigienvalues*1.0/total<=threshold):
		k_eigienvalues+=eigenvalues[k]
		new_eigenvectors.append(eigenvectors[:,k])
		k+=1
	# print k
	new_eigenvectors = np.array(new_eigenvectors)
	return new_eigenvectors
