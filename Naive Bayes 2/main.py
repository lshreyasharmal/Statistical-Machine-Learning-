import mnist
from mnist import MNIST
import random
from train_model import train_
from test_model import test_
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import numpy as np

mndata = MNIST('shreya')

images, labels = mndata.load_training()
# or
images1, labels1 = mndata.load_testing()


# index = random.randrange(0, len(images))  # choose an index ;-)
# print(mndata.display(images[index]))

low = [0,3]
high = [1,8]


for yo in range(2):
	C1 = low[yo]
	C2 = high[yo]
	print str(C1) + " " + str(C2)
	if(C1 == 0 and C2== 1):
		train_data=[]
		test_data=[]
		train_labels=[]
		test_labels=[]


		for i in range(len(labels)):
			if(labels[i]==C2 or labels[i]==C1):
				train_data.append(images[i])
				train_labels.append(labels[i])


		for i in range(len(labels1)):
			if(labels1[i]==C2 or labels1[i]==C1):
				test_data.append(images1[i])
				test_labels.append(labels1[i])

		model_low, model_high, p1, p2 = train_(train_data,train_labels,C1,C2)
		# print model_low
		print
		# print model_high
		TPR, FAR, ACC, fpr, tpr,thresholds = test_(test_data,test_labels,model_low,model_high,p1,p2,C1,C2)
		roc_auc = metrics.auc(fpr,tpr)
		plt.figure()
		plt.plot(fpr,tpr,color='r',label='ROC curve (area=%0.2f'%roc_auc)
		plt.plot([0,1],[0,1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend(loc ='lower right')
		plt.title("ROC CURVE")
		plt.show()
		print 
	elif(C1 == 3 and C2 == 8):
		class3 = []
		class3_labels = []
		class8 = []
		class8_labels = []

		for i in range(len(labels)):
			if(labels[i]==C1):
				class3.append(images[i])
				class3_labels.append(labels[i])
			elif(labels[i]==C2):
				class8.append(images[i])
				class8_labels.append(labels[i])
		class3_test = []
		class3_labels_test = []
		class8_test = []
		class8_labels_test = []

		for i in range(len(labels1)):
			if(labels1[i]==C1):
				class3_test.append(images1[i])
				class3_labels_test.append(labels1[i])
			elif(labels1[i]==C2):
				class8_test.append(images1[i])
				class8_labels_test.append(labels1[i])

		train_data=[]
		train_labels=[]
		test_data=[]
		test_labels=[]

		n1 = random.sample(range(len(class3)), int(0.1*len(class3)))
		n2 = random.sample(range(len(class8)), int(0.9*len(class8)))
		m1 = random.sample(range(len(class3_test)), int(0.1*len(class3_test)))
		m2 = random.sample(range(len(class8_test)), int(0.9*len(class8_test)))


		for i in range(len(class3)):
			if i in n1:
				train_data.append(class3[i])
				train_labels.append(class3_labels[i])
		for i in range(len(class8)):
			if i in n2:
				train_data.append(class8[i])
				train_labels.append(class8_labels[i])
		for i in range(len(class3_test)):
			if i in m1:
				test_data.append(class3_test[i])
				test_labels.append(class3_labels_test[i])
		for i in range(len(class8_test)):
			if i in m2:
				test_data.append(class8_test[i])
				test_labels.append(class8_labels_test[i])

		model_low, model_high, p1, p2 = train_(train_data,train_labels,C1,C2)
		# print model_high
		# print 
		# print model_low
		TPR, FAR, ACC, fpr, tpr,thresholds = test_(test_data,test_labels,model_low,model_high,p1,p2,C1,C2)
		roc_auc = metrics.auc(fpr,tpr)
		plt.figure()
		plt.plot(fpr,tpr,color='r',label='ROC curve (area=%0.2f'%roc_auc)
		plt.plot([0,1],[0,1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend(loc ='lower right')
		plt.title("ROC CURVE")
		plt.show()

		print



