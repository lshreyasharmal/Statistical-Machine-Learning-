import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import metrics
import random
from scipy import misc
from scipy.misc import imread
from skimage.transform import rescale, resize, downscale_local_mean
import os
from train_PCA import train_pca
from test_LDA import test_
from test_PCA import test_pca
from train_LDA import LDA
import cPickle
from skimage.measure import block_reduce
from tempfile import TemporaryFile

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

whichdata = 2
data = []
labels = []
des = 0
if(whichdata == 1):
	for k in range(1,6):
		folder = 'cifar-10-batches-py/data_batch_'+str(k)
		dic = unpickle(folder)
		data.extend(dic['data'])
		labels.extend(dic['labels'])
	folder = 'cifar-10-batches-py/test_batch'
	dic = unpickle(folder)
	data.extend(dic['data'])
	labels.extend(dic['labels'])
	print len(data)
	print len(labels)
	data = np.array(data)
	print data.shape
	new_data=[]
	data = np.array(data)
	des = data.shape[0]/100
if(whichdata==2):
	data = []
	labels = []
	for i in range(1,12):
		folder = "./Face_data/"+str(i)+"/"
		for face in os.listdir(folder):
			if("yale" in face and "info" not in face):
				f = misc.imread(folder+face)
				f=downscale_local_mean(f,(12,12))
				if(f.shape==(16,14)):
					data.append(f.ravel())
					labels.append(i)
	data = np.array(data)
	labels = np.array(labels)
	print data.shape
	des = data.shape[0]

data = np.array(data)
acc = []
FPRS = []
TPRS = []
models = []
for _ in range(5):
	i_train = random.sample(range(0, data.shape[0]), int(0.7*(data.shape[0])))


	Xtest = []
	Ytest = []
	Xtrain = []
	Ytrain = []
	print data.shape[0]
	size = des
	for i in range(0,size):

		if i in i_train:
			Xtrain.append(data[i])
			Ytrain.append(labels[i])
		else:
			Xtest.append(data[i])
			Ytest.append(labels[i])

	Xtest = np.array(Xtest)
	Xtrain = np.array(Xtrain)
	Ytest = np.array(Ytest)
	Ytrain = np.array(Ytrain)
	# print Xtrain.shape
	# print Xtest.shape

	Xtrai,Xtes = train_pca(Xtrain,Xtest)
	#FOR LDA
	# eigenvectors = np.array([])
	# if(whichdata==1):
	# 	eigenvectors = LDA(Xtrain,Ytrain,10)
	# else:
	# 	eigenvectors = LDA(Xtrain,Ytrain,11)
	# models.append(eigenvectors)
	

	# print eigenvectors.shape

	# Xtrai = Xtrain.dot(eigenvectors.T)
	# print Xtrai.shape
	# Xtes = Xtest.dot(eigenvectors.T)


	# FOR PCA
	
	f = []
	f.append(Xtrai)
	f.append(Xtes)
	f=np.array(f)
	models.append(f)


	# test if PCA IS RECENT
	a,tpr,fpr=test_pca(Xtrai,Ytrain,Xtes,Ytest)

	#test if LDA is recent
	# a,tpr,fpr=test_(Xtrai,Ytrain,Xtes,Ytest)

	TPRS.append(tpr)
	FPRS.append(fpr)
	acc.append(a)
	# acc.append(test_(Xtrain,Ytrain,Xtest,Ytest))
models = np.array(models)
outfile = open("PCAface.npy","w+")
np.save(outfile, models)
print acc
acc = np.array(acc)
print "mean"
print np.mean(acc)
print "stddev"
print np.std(acc)
TPRS = np.array(TPRS)
FPRS = np.array(FPRS)
xx = np.mean(TPRS,axis=0)
yy = np.mean(FPRS,axis=0)

plt.plot(yy,xx,"o",label = "Roc Curve")
plt.plot([0,1],[0,1])
plt.xlabel("False Acceptance Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC CURVE")
plt.legend(loc='upper left')
plt.show()
