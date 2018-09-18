import numpy as np
import math
from numpy import linalg as LA
from numpy.linalg import inv


def LDA(X,Y,l):
	classes = []

	X = np.array(X)
	Y = np.array(Y)

	for j in range(l):
		t = []
		for i in range(len(X)):
			if(l==10):
				if Y[i]==j:
					t.append(X[i])
			if(l==11):
				if Y[i]==j+1:
					t.append(X[i])
		t = np.array(t)
		classes.append(t)
	classes = np.array(classes)
	print classes.shape
	means = []
	for cl in classes:
		# print cl.shape
		means.append(np.mean(cl,axis=0))
	means = np.array(means)
	# print means[9]
	d = X[0].shape[0]
	Sw = np.zeros((d,d))
	for i in range(len(classes)):
		cov = np.matmul((classes[i]-means[i]).T,(classes[i]-means[i]))

		print cov.shape
		# print cov
		# print cov.shape
		# cov = np.multiply(len(classes[i])-1,cov)
		Sw = np.add(Sw,cov)
	print Sw.shape
	print  LA.det(Sw)
	Sb = np.cov(X.T)-Sw
	print Sb.shape
	# print d*d
	# nu = np.mean(X,axis=0)
	# for j in range(len(classes)):
	# 	x = np.subtract(means[j],nu)
	# 	x = np.atleast_2d(x)
	# 	xx = np.dot(x,x.T)
	# 	x3 = np.multiply(len(classes[j]),xx)
	# 	# print xx
	# 	Sb=np.add(x3,Sb)
	Sw_i = inv(Sw)
	# print Sw_i
	# print Sb.shape
	# # print Sb
	# # print Sw_i
	# # print Sw_i.dot(Sb.T)
	ev,ew = np.linalg.eigh(Sb.dot(Sw_i))
	ev = ev.real
	ew = ew.real
	print "j"
	print ev
	# # print X.shape
	# # print ew.shape

	# # eigenvectors = np.dot(X,ew)
	# # print len(eigenvectors)
	# # for i in range(len(X)):
	# 	# eigenvectors[:,i]/=np.linalg.norm(eigenvectors[:,i])
	# # np.linalg.norm(u[:,i])
	eigenvalues = ev 
	# ev = np.absolute(ev)
	# ew = np.absolute(ew)
	# idx = eigenvalues.argsort()[::-1]  
	# eigenvalues = eigenvalues[idx]
	# ew = ew[:,idx]

	eigenpairs = [(np.array(eigenvalues[i]), ew[:,i]) for i in range(len(eigenvalues))]
	eigenpairs.sort(key=lambda x: x[0], reverse=True)	
	print "kk"
	print ev
	eigenpairs_top=[]
	for i in range(l-1):
		print eigenpairs[i][0]
		eigenpairs_top.append(eigenpairs[i][1])
	# print ew.shape
	# # for r in eigenvalues:
	# # 	print r
	# new_eigenvectors = []
	# for c in range(len(ew)):
	# 	if c==l-1:
	# 		break
	# 	new_eigenvectors.append(ew[:,c])
	new_eigenvectors=np.array(eigenpairs_top)
	print new_eigenvectors	
	print "TRAIN DONE"
	return new_eigenvectors