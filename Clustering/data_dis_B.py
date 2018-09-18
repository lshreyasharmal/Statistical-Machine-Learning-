import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score
n = 1600
centre = [1,2]
radius = 1
angle=[]
for i in range(n):
	angle.append(2*math.pi*random.random())
r = []
for i in range(n):
	r.append(radius*math.sqrt(random.random()))
X1 = []
Y1 = []

for i in range(n):
	X1.append(r[i]*math.cos(angle[i])+centre[0])
	Y1.append(r[i]*math.sin(angle[i])+centre[1])
X1=np.array(X1)
Y1=np.array(Y1)
l1 = np.ones(len(Y1))

n = 1600
centre = [3,4]
radius = 2
angle=[]
for i in range(n):
	angle.append(2*math.pi*random.random())
r = []
for i in range(n):
	r.append(radius*math.sqrt(random.random()))
X2 = []
Y2 = []

for i in range(n):
	X2.append(r[i]*math.cos(angle[i])+centre[0])
	Y2.append(r[i]*math.sin(angle[i])+centre[1])
X2=np.array(X2)
Y2=np.array(Y2)
l2 = np.zeros(len(Y2))

X = []
X.append(np.array(X1))
X.append(np.array(X2))
Y = []
Y.append(np.array(Y1))
Y.append(np.array(Y2))

colors = ['bo','ro']
for i in range(2):
	plt.plot(X[i],Y[i],colors[i])
plt.title("Data B with radius 2")
plt.savefig("dataB_2")
plt.show()

X1= np.array(X1)
X2 = np.array(X2)
Y1= np.array(Y1)
Y2 = np.array(Y2)
X = np.concatenate((X1,X2),axis=0)

Y = np.concatenate((Y1,Y2),axis=0)
L = np.concatenate((l1,l2),axis=0)
print X.shape
print Y.shape
data = np.column_stack((X,Y))
data = StandardScaler().fit_transform(data)
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
y_means = kmeans.predict(data)
print "Accuracy : " + str(accuracy_score(L,y_means))
print y_means
colors = ['bo','ro']
for i in range(len(X)):
	plt.plot(X[i],Y[i],colors[y_means[i]])
plt.title("K-means Clustering on Data B with radius 2")
plt.savefig("Data_B_kmeans2")
plt.show()
connectivity = kneighbors_graph(data, n_neighbors=10, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)
model = AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=2, connectivity=connectivity)
model.fit_predict(data)
y = model.labels_
print y
print "Accuracy : " + str(accuracy_score(L,y))
colors = ['bo','ro']
for i in range(len(X)):
	plt.plot(X[i],Y[i],colors[y[i]])
plt.title("Hierarchical Clustering on Data B with radius 2")
plt.savefig("Data_B_hierarchical2")
plt.show()
