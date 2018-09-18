import numpy as np
import os
from numpy import linalg as LA
from scipy import misc
import matplotlib.pyplot as plt
import random
from sklearn.naive_bayes import GaussianNB
from pca import pca
from sklearn.decomposition import PCA

def classify(trainX,trainY,testX,testY):

	# print trainX.shape
	# print trainY.shape
	# print testX.shape
	# print testY.shape
	# return error
	clf = GaussianNB()
	clf.fit(trainX, trainY)
	result = clf.predict(testX)
	# print result
	e = 0
	for i in range(len(result)):
		if result[i]==testY[i]:
			e+=1
	return (e*100.0/len(result))


train_set = []
train_labels = []

test_set = []
test_labels = []

for i in range(1,12):
	folder = "./Face_data/"+str(i)+"/"
	temp = []
	j=0
	test_indices = random.sample(range(65), int(0.5*65))
	for face in os.listdir(folder):
		if("yale" in face and "info" not in face):
			f = misc.imread(folder+face)
			if(j in test_indices):
				test_set.append(f)
				test_labels.append(i)
			else:
				train_set.append(f)
				train_labels.append(i)
			j+=1
train_set = np.array(train_set)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
test_set = np.array(test_set)



reshaped_images = []
for i in train_set:
	x = np.resize(i,(192,168))
	f = x.shape
	x = x.reshape(f[0]*f[1])
	reshaped_images.append(x)
reshaped_images = np.array(reshaped_images)

train_set = reshaped_images

reshaped_images = []
for i in test_set:
	x = np.resize(i,(192,168))
	f = x.shape
	x = x.reshape(f[0]*f[1])
	reshaped_images.append(x)
reshaped_images = np.array(reshaped_images)

test_set = reshaped_images
# print test_set.shape
# print train_set.shape

print "Accuracy without PCA :"
print classify(train_set,train_labels,test_set,test_labels)
images = np.concatenate((train_set,test_set),axis=0)

# # Compute a PCA 
# n_components = 184
# pp = PCA(n_components=n_components, whiten=True).fit(images)
 
# # apply PCA transformation
# X_train_pca = pp.transform(train_set)
# X_test_pca = pp.transform(test_set)

# print classify(X_train_pca,train_labels,X_test_pca,test_labels)
images = np.concatenate((train_set,test_set),axis=0)
# print images.shape
new_eigenvectors = pca(0.99,images)
# print test_set.shape
# print "==="
test_set = np.dot(test_set,new_eigenvectors.T)
# print test_set.shape

train_set = np.dot(train_set,new_eigenvectors.T)
print train_set.shape
print "Accuracy with PCA : "
print classify(train_set,train_labels,test_set,test_labels)
