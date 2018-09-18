import math
import numpy as np
from scipy import integrate
from numpy.linalg import solve
import matplotlib.pyplot as plt
def discriminant_coeff(mean,variance,prior):
	# print variance
	a2 = (-1.0)/(2*variance)
	a1 = (mean*1.0)/variance
	a0 = math.log(prior)-0.5*(math.log(2*math.pi))-0.5*(math.log(variance))-0.5*((math.pow(mean,2)*1.0)/(variance))
	return a2,a1,a0

def solve(mean,variance,priors):
	a2,a1,a0 = discriminant_coeff(mean[0],variance[0],priors[0])
	b2,b1,b0 = discriminant_coeff(mean[1],variance[1],priors[1])
	c1 = a2-b2
	c2 = a1-b1
	c3 = a0-b0
	coeff = [c1,c2,c3]
	return np.roots(coeff)

# def erf1():

def erf(mean,variance,priors,x1,x2):
	if(x1==x2):
		variance[0]=math.sqrt(variance[0])
		variance[1]=math.sqrt(variance[1])
		constant1 = priors[0]/((math.sqrt(2*math.pi))*(variance[0]))
		constant2 = priors[1]/((math.sqrt(2*math.pi))*(variance[1]))
		func1 = lambda x: constant1*np.exp( -0.5*(math.pow(( (x-mean[0])*1.0/(variance[0]) ),2)) )
		func2 = lambda x: constant2*np.exp( -0.5*(math.pow(( (x-mean[1])*1.0/(variance[1]) ),2)) )
		val1 = integrate.quad(func1,x2,np.inf)
		val2 = integrate.quad(func2,-np.inf,x1)
		error = val1[0]+val2[0]
		return error
	else:
		variance[0]=math.sqrt(variance[0])
		variance[1]=math.sqrt(variance[1])
		constant1 = priors[0]/((math.sqrt(2*math.pi))*(variance[0]))
		constant2 = priors[1]/((math.sqrt(2*math.pi))*(variance[1]))
		#ERF FUNCTION : func1 and func 2
		func1 = lambda x: constant1*np.exp( -0.5*(math.pow(( (x-mean[0])*1.0/(variance[0]) ),2)) )
		func2 = lambda x: constant2*np.exp( -0.5*(math.pow(( (x-mean[1])*1.0/(variance[1]) ),2)) )
		val1 = integrate.quad(func1,x2,np.inf)
		val2 = integrate.quad(func1,-np.inf,x1)
		val3 = integrate.quad(func2,x1,x2)
		error = val1[0]+val2[0]+val3[0]
		return error


def discriminant_val(mean,variance,prior,x):
	a2,a1,a0 = discriminant_coeff(mean,variance,prior)
	# print a2*pow(x,2)+a1*x+a0
	return a2*pow(x,2)+a1*x+a0

def classify(s1,s2,x,priors):
	mean = []
	variance = []
	mean.append(np.mean(s1))
	mean.append(np.mean(s2))
	variance.append(np.var(s1))
	variance.append(np.var(s2))
	d1 = discriminant_val(mean[0],variance[0],priors[0],x)
	d2 = discriminant_val(mean[1],variance[1],priors[1],x)
	if d1 >= d2:
		return 1
	else:
		return 2

def bhattacharyyabound(mean,variance,priors):

	nu1 = mean[0]
	nu2 = mean[1]
	E1 = variance[0]
	E2 = variance[1]
	E1E2 = (E1+E2)*0.5
	p1 = priors[0]
	p2 = priors[1]
	value = (1.0/8)*(math.pow((nu1-nu2),2)/float(E1E2)) + 0.5*(math.log(E1E2/(math.sqrt(abs(E1*E2)))))
	return math.sqrt(p1)*math.sqrt(p2)*math.exp(-1*value)

mean=[[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]]
variance=[[1,1],[2,2],[2,2],[3,1]]
priors=[[0.5,0.5],[2.0/3,1.0/3],[0.5,0.5],[0.5,0.5]]

T = range(10,1001,10)
T=np.array(T)
L = [10,30,100,200,500,1000]
for i in range(4):

		print "Bhattacharyya Bound for mean = " + str(mean[i])+ " varia	nce = " + str(variance[i]) + " and prior = " + str(priors[i]) + " : " + str(bhattacharyyabound(mean[i],variance[i],priors[i]))

		x =  solve(mean[i],variance[i],priors[i])
		x1 = 0
		x2 = 0
		if(len(x)==1):
			x1 = x[0]
			x2 = x[0]
		else:
			x1 = x[0]
			x2 = x[1]
		print x
		print "True Error in terms of erf for = " + str(mean[i])+ " variance = " + str(variance[i]) + " and prior = " + str(priors[i]) + " : "  + str(erf(mean[i],variance[i],priors[i],x1,x2))

		# print 0.5*(math.erf((x1*(math.sqrt(variance[i][0])+mean[i][0]))/math.sqrt(2)) - math.erf((x1*(math.sqrt(variance[i][1])+mean[i][1]))/math.sqrt(2)))

		Error = []
		E = []
		for t in T:
			s1 = np.random.normal(mean[i][0], variance[i][0], t)
			s2 = np.random.normal(mean[i][1], variance[i][1], t)
			S = np.concatenate((s1,s2),axis=0)
			o1 = []
			for s in S:
				o1.append(classify(s1,s2,s,priors[i]))
			# print o1
			e = 0

			for j in range(t):
				if o1[j]!=1:
					e+=1
				if o1[j+t]!=2:
					e+=1
			Error.append(e/(2.0*t))
			if(t in L):
				E.append(e/(2.0*t))
				print "Empirical error for "+ str(t)+" random points "+ str(e/(2.0*t))

		# print Error
		print 
		print 

		# plt.plot(range(len(T)),Error,'ro',color='r',label='Error')
		# plt.axis([0,len(T),0,1])
		# plt.xlabel('Number of samples')
		# plt.ylabel('Error')
		# plt.xticks(np.arange(len(L)), L)
		# plt.legend(loc ='lower right')
		# plt.show()
		plt.plot(range(len(L)),E,'ro',color='r',label='Error')
		plt.axis([0,len(L),0,1])
		plt.xlabel('Number of samples')
		plt.ylabel('Error')
		plt.xticks(np.arange(len(L)), L)
		plt.legend(loc ='lower right')
		plt.title("Error vs Number of data points with " + str(mean[i])+ " variance = " + str(variance[i]) + " and prior = " + str(priors[i]) )
		plt.show()
		# break
