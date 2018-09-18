import numpy as np
from numpy.linalg import inv
import math


def mahalanobisdistance(point,data):

	mean = np.mean(data,axis=0)
	sigma =  np.cov(data.T)
	inverse = np.linalg.inv(sigma)
	x = np.subtract(point,mean)
	return math.pow((np.dot(x.T,np.dot(inverse,x))),0.5)
def classify(point,w1,w2,w3,p1,p2,p3):

	w1 = np.array(w1)
	w2 = np.array(w2)
	w3 = np.array(w3)
	sigma1 = np.cov(w1.T)
	sigma2 = np.cov(w2.T)
	sigma3 = np.cov(w3.T)
	m1 = mahalanobisdistance(point,w1) 
	m2 = mahalanobisdistance(point,w2)
	m3 = mahalanobisdistance(point,w3)
	# print m1
	# print m2
	# print m3
	# print sigma1
	# print sigma2
	# print sigma3
	print point
	print "mahalanobis distance from w1 = "+str(m1)
	print "mahalanobis distance from w2 = "+str(m2)
	print "mahalanobis distance from w3 = "+str(m3)

	m1 = -0.5*pow(m1,2)
	m2 = -0.5*pow(m2,2)
	m3 = -0.5*pow(m3,2)

	m1 += np.log(p1) -0.5*np.log(np.linalg.det(sigma1))
	m2 += np.log(p2) -0.5*np.log(np.linalg.det(sigma2))
	m3 += np.log(p3) -0.5*np.log(np.linalg.det(sigma3))

	# print m1
	# print m2
	# print m3

	if(m1>=m2 and m1>=m3):
		return 1
	elif(m2>=m1 and m2>=m3):
		return 2
	elif(m3>=m1 and m3>=m2):
		return 3




points = [[1,2,1]
,[5,3,2]
,[0,0,0]
,[1,0,0]]

w1=[[-5.01,-8.12,-3.68]
,[-5.43,-3.48,-3.54]
,[1.08,-5.52,1.66]
,[0.86,-3.78,-4.11]
,[-2.67,0.63,7.39]
,[4.94,3.29,2.08]
,[-2.51,2.09,-2.59]
,[-2.25,-2.13,-6.94]
,[5.56,2.86,-2.26]
,[1.03,-3.33,4.33]]

w2=[[-0.91,-0.18,-0.05]
,[1.3,-2.06,-3.53]
,[-7.75,-4.54,-0.95]
,[-5.47,0.5,3.92]
,[6.14,5.72,-4.85]
,[3.6,1.26,4.36]
,[5.37,-4.63,-3.65]
,[7.18,1.46,-6.66]
,[-7.39,1.17,6.3]
,[-7.5,-6.32,-0.31]]

w3=[[5.35,2.26,8.13]
,[5.12,3.22,-2.66]
,[-1.34,-5.31,-9.87]
,[4.48,3.42,5.19]
,[7.11,2.39,9.21]
,[7.17,4.33,-0.98]
,[5.75,3.97,6.65]
,[0.77,0.27,2.41]
,[0.9,-0.43,-8.71]
,[3.52,-0.36,6.43]]

print "With equal priors"
print
output = []
p1 = 1.0/3
p2 = 1.0/3
p3 = 1.0/3
for p in points:
	output.append(classify(p,w1,w2,w3,p1,p2,p3))
print "RESULT"
print output
print "----------------------------"

print
print "With priors 0.8, 0.1 and 0.1"
print 
output = []
p1 = 0.8
p2 = 0.1
p3 = 0.1
for p in points:
	output.append(classify(p,w1,w2,w3,p1,p2,p3))
print "RESULT"
print output
