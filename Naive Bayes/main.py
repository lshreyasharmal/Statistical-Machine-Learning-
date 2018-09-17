import numpy as np
import random

from ttrain_bayes import *
from ttest_bayes import testt
np.random.seed(4)
def func1():
	# open dataset
	with open('dataset.txt','r') as dataset:
		input = dataset.readlines()


	# create i/o
	x=[]
	y=[]
	for line in input:
		temp = line.strip().split(",")
		if temp[-1] != '2':
			y.append(temp[-1])
			x.append(temp[:5])
	x = np.array(x)
	y = np.array(y)


	# split into test and train
	test = []
	train = []
	testy=[]
	trainy=[]
	train_indices = random.sample(range(len(x)), int(0.7*len(x)))
	num_low = 0
	num_high = 0
	for i in range(len(x)):
		if i in train_indices:
			train.append(x[i])
			trainy.append(y[i])
			if(y[i]=='1'):
				num_low+=1
			else:
				num_high+=1
		else:
			test.append(x[i])
			testy.append(y[i])
	p1 = num_low*1.0/(num_low+num_high)
	p2 = num_high*1.0/(num_high+num_low)
	test = np.array(test)
	train = np.array(train)
	feature1, feature2, feature3, feature4, feature5, test, testy, train, trainy, num_high, num_low = func2(test, train, num_low, num_high, trainy,testy)
	num_high, num_low, acc, likelihood_low_arr, likelihood_high_arr = testt(feature1, feature2, feature3, feature4, feature5, test, testy, train, trainy, num_high, num_low)
	print "Likelihood Values of Class Low"
	print likelihood_low_arr

	print "Likelihood Values of Class High"
	print likelihood_high_arr

	print "Posterior*Evidence Values of Class Low"
	print  [p1*_ for _ in likelihood_low_arr]

	print "Posterior*Evidence Values of Class High"
	print  [p2*_ for _ in likelihood_high_arr]

	print "alright"
def cross_validation_split(datasetX,datasetY, folds=5):
	datasetX_split = []
	datasetY_split = []
	fold_size = int(len(datasetX)/folds)
	idx = 0
	for i in range(folds):
		foldX = []
		foldY = []
		for j in range(idx,idx+fold_size):
			foldX.append(datasetX[j])
			foldY.append(datasetY[j])
		idx = idx+fold_size
		datasetX_split.append(foldX)
		datasetY_split.append(foldY)
	return datasetX_split, datasetY_split

def kfold():
	with open('dataset.txt','r') as dataset:
		input = dataset.readlines()


	# create i/o
	x=[]
	y=[]
	for line in input:
		temp = line.strip().split(",")
		if temp[-1] != '2':
			y.append(temp[-1])
			x.append(temp[:5])
	x = np.array(x)
	y = np.array(y)


	# split into 5 folds


	num_folds = 5

	resX, resY = cross_validation_split(x,y,5)

	Accuracies = []

	for i in range(num_folds):
		testX = resX[i]
		trainX =[]
		testY = resY[i]
		trainY =[]
		for k in range(num_folds):
			if(k!=i):
				trainX+=resX[k]
				trainY+=resY[k]
		
		num_low = 0
		num_high = 0 

		for k in range(len(trainX)):
			if(trainY[k]=='1'):
				num_low+=1
			else:
				num_high+=1
		feature1, feature2, feature3, feature4, feature5, test, testy, train, trainy, num_high, num_low = func2(testX, trainX, num_low, num_high, trainY,testY)

		num_high, num_low, acc, likelihood_low_arr, likelihood_high_arr = testt(feature1, feature2, feature3, feature4, feature5, test, testy, train, trainy, num_high, num_low)
		Accuracies.append(acc)
	print "Accuracies " + str(Accuracies)
	Accuracies = np.array(Accuracies)
	print "Mean Accuracy = %f"%(np.average(Accuracies)) 
	print "Standard Deviation = %f"%(np.std(Accuracies)) 

# kfold()

func1()
