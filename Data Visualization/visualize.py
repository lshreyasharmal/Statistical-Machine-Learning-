import numpy as np
import matplotlib.pyplot as plt
import math
np.random.seed(1) #For reproducibility 
mu = [2,3] 
sigma = [[1,1.5],[1.5,30]]
#Randomly Generated 100 Points
x, y = np.random.multivariate_normal(mu, sigma, 100).T 
# print x 
# print y

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Dataset of 100 samples")

plt.plot(x, y, 'x', label = 'data points')
plt.legend(loc='upper left')
plt.show()

covariance_matrix =  np.corrcoef(x,y)
corr = covariance_matrix[0][1]
print "The Correlation of the 2D dataset is: " + str(corr)
print "----------------------------------------------------------"


print "Varying the sigma parameter for Y we observe that Correlation decreases as sigma parameter of y increases"
corrs = []
ys = []
for i in range(500):
	mu = [2,3] 
	sigma = [[1,1.5],[1.5,i]]
	x, y = np.random.multivariate_normal(mu, sigma, 100).T

	covariance_matrix =  np.corrcoef(x,y)
	corr = covariance_matrix[0][1]
	corrs.append(corr)
	ys.append(i)
# print corrs

plt.plot(ys,corrs,'o', label = 'Correlation Values')
plt.xlabel("Sigma parameter of y")
plt.ylabel("Correlation of x,y")
plt.title("Effect of sigma parameter variation on Correlation of x & y")
plt.legend(loc='upper left')
plt.show()


print "---------------------------------------"


mu = [0,0] 
sigma = [[1,0.9],[0.9,1]]
x, y = np.random.multivariate_normal(mu, sigma, 1200).T
plt.plot(x, y, 'o', label = "data points")
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("Diagonally distributed data with Correlation = 0.9")
plt.legend(loc='upper left')
plt.show()
d, E = np.linalg.eigh(sigma)
EPS = 10e-5
D = np.diag(1. / np.sqrt(d + EPS))
W = np.dot(np.dot(E, D), E.T)
X = np.dot(np.asarray([x,y]).T,W)


plt.plot(X[:,0],X[:,1], "o", label = "data points")
plt.xlabel("x values, Correlation = " + str(np.corrcoef(X[:,0],X[:,1])[0][1]))
plt.ylabel("y values")
plt.title("Data Decorrelated after performing whitening transformation")
plt.legend(loc='upper left')
plt.show()

print  "Yes we can decorrelate the data. The Correlation after Decorrelation (Whitening) "+str(np.corrcoef(X[:,0],X[:,1])[0][1])


print "------------------------------------------------------------"
mu = [2,3] 
sigma = [[1,1.5],[1.5,30]]
X, Y = np.random.multivariate_normal(mu, sigma, 100).T #Randomly Generated 100 Points
# print x 
# print y

X = np.multiply(X,X)
Y = np.multiply(Y,Y)
plt.plot(X, Y, 'x', color = 'orange', label = 'data points')
plt.xlabel("x2 values")
plt.ylabel("y2 values")
plt.title("Data points of x^2 and y^2")
plt.legend(loc='upper left')
plt.show()

covariance_matrix =  np.corrcoef(X,Y)
corr = covariance_matrix[0][1]
print "The new Correlation of the 2D dataset of x^2 and y^2 is: " + str(corr)

print "---------------------------------------------"
mu = [-2,3] 
sigma = [[1,1.5],[1.5,30]]
x, y = np.random.multivariate_normal(mu, sigma, 100).T #Randomly Generated 100 Points
# print x 
# print y
covariance_matrix =  np.corrcoef(x,y)
corr1 = covariance_matrix[0][1]
plt.plot(x,y, "o", label = 'data points x,y')
plt.xlabel("X Values, X2 Values")
plt.ylabel("Y Values, Y2 Values")
plt.title("Dataset of 100 samples with mu = [-2,3]")

X = np.multiply(x,x)
Y = np.multiply(y,y)
plt.plot(X, Y, 'x', label = 'data points x2,y2')
plt.legend(loc='upper left')
plt.show()

covariance_matrix =  np.corrcoef(X,Y)
corr = covariance_matrix[0][1]
print "The Correlation of the 2D dataset (x,y) is: " + str(corr) + " and that of (x2,y2) is: " + str(corr1)
