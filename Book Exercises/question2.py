import numpy as np
import math
from numpy.linalg import inv

def discriminant_val(train,test,d,p):
	if(d==1):
		mean = np.mean(train)
		sigma = np.cov(train.T)
		x = test-mean
		value = -0.5*np.log(2*math.pi) - 0.5*np.log(sigma) - 0.5*((math.pow(x,2))*1.0/sigma) + np.log(p)
		
		return value
	mean = np.mean(train,axis=0)

	sigma = np.cov(train.T)
	
	x = np.subtract(test,mean)

	inverse = np.linalg.inv(sigma)

	value = -0.5*d*math.log(2*math.pi) - 0.5*math.log(np.linalg.det(sigma)) -0.5*(np.dot(np.dot(x,inverse),x))

	return value




def classify(train1,train2,test,d,p):
	class1 = []
	class2 = []

	for i in range(len(train1)):
		x = []
		xx = []
		for dd in range(d):
			x.append(train1[i][dd])
			xx.append(train2[i][dd])
		class1.append(x)
		class2.append(xx)
	class1 = np.array(class1)
	class2 = np.array(class2)

	if(discriminant_val(class1,test,d,p)-discriminant_val(class2,test,d,p)>=0):
		return 1
	else:
		return 2

def bhattacharya(train1,train2,d,p1,p2):
	class1 = []
	class2 = []

	for i in range(len(train1)):
		x = []
		xx = []
		for dd in range(d):
			x.append(train1[i][dd])
			xx.append(train2[i][dd])
		class1.append(x)
		class2.append(xx)
	class1 = np.array(class1)
	class2 = np.array(class2)


	nu1 = np.mean(class1,axis=0)
	nu2 = np.mean(class2,axis=0)

	if d==1:
		E1 = (np.var(class1))
		E2 = (np.var(class2))
	
		E1E2 = 0.5*np.add(E1,E2)

		value = (1.0/8)*(math.pow((nu1-nu2),2)/float(E1E2)) + 0.5*(math.log(E1E2/(math.sqrt(abs(E1*E2)))))


		return math.sqrt(p1)*math.sqrt(p2)*math.exp(-1*value)
	else:
		E1 = np.cov(class1.T)
		E2 = np.cov(class2.T)

	det_E1 = np.linalg.det(E1)
	det_E2 = np.linalg.det(E2)

	E1E2 = 0.5*np.add(E1,E2)
	E1E2_inv = np.linalg.inv(E1E2)
	E1E2_det = np.linalg.det(E1E2)

	value = (1.0/8)*(np.dot( np.dot( np.subtract(nu1,nu2),E1E2_inv),np.subtract(nu1,nu2)))+0.5*(np.log(E1E2_det)/math.sqrt(det_E1*det_E2))
	return math.sqrt(p1)*math.sqrt(p2)*math.exp(-1*value)




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


for d in range(1,4):
	question1 = []
	e1 = 0
	for i in range(len(w1)):
		x = []
		for k in range(d):
			x.append(w1[i][k])
		question1.append(classify(w1,w2,x,d,0.5))

	for q in question1:
		if q!=1:
			e1+=1
	# print e/10.0
	print "Result for 10 samples of w1 :"
	print question1
	print "Error for 10 samples of w1: " + str(e1/10.0)
	question2 = []
	e2=0
	for i in range(len(w2)):
		x = []
		for k in range(d):
			x.append(w2[i][k])
		question1.append(classify(w1,w2,x,d,0.5))
		question2.append(classify(w1,w2,x,d,0.5))


	for i in range(len(question2)):
		if question2[i]!=2:
			e2+=1
	print "Result for 10 samples of w2 :"
	print question2
	print "Error for 10 samples of w2: " + str(e2/10.0)

	print "OVERALL ERROR (20 samples) : " + str((e1+e2)/20.0)

	print "bhattacharya bound : " + str(bhattacharya(w1,w2,d,0.5,0.5))
	print
	print 
