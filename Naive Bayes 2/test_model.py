import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import metrics

def test_(test_data,test_labels,model_low,model_high,p1,p2,C1,C2):

	for i in range(len(test_data)):
		for x in range(len(test_data[i])):
			test_data[i][x]/=255.0
	for i in range(len(test_data)):
		for x in range(len(test_data[i])):
			if test_data[i][x]>0.5:
				test_data[i][x]=1
			else:
				test_data[i][x]=0

	output = []
	prob_labels = []
	for i in range(len(test_data)):
		likelihood_low = 0
		likelihood_high = 0
		for k in range(784):
			likelihood_low+=math.log(model_low[k][test_data[i][k]])
			likelihood_high+=math.log(model_high[k][test_data[i][k]])

			# if(likelihood_low<0):
			# 	print "haww"
			# if likelihood_high<0:
			# 	print"omg"

		prob_labels.append(likelihood_high+math.log(p2))
		if(likelihood_low + math.log(p1) > likelihood_high+ math.log(p2)):
			output.append(C1)
		else:
			output.append(C2)
	TP=0
	TN=0
	FP=0
	FN=0
 
	for i in range(len(test_labels)):
		if(test_labels[i]==C1 and output[i]==C1):
			TP+=1
		elif(test_labels[i]==C2 and output[i]==C1):
			FP+=1
		elif(test_labels[i]==C1 and output[i]==C2):
			FN+=1
		elif(test_labels[i]==C2 and output[i]==C2):
			TN+=1

	FAR = FP*1.0/(FP+TN)
	TPR = TP*1.0/(TP+FN)
	ACC = (TP+TN)*1.0/(TP+FP+TN+FN)
	print "FAR " + str(FAR)
	print "TPR " + str(TPR)
	print "ACC " +str(ACC)
	# print A
	# print B
	# print C
	# print D
	# print output
	fpr, tpr, threasholds = metrics.roc_curve(test_labels,prob_labels,pos_label=C2)
	# roc_auc = metrics.auc(fpr,tpr)
	# plt.figure()
	# plt.plot(fpr,tpr,color='r',label='ROC curve (area=%0.2f'%roc_auc)
	# plt.plot([0,1],[0,1])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.legend(loc ='lower right')
	# plt.title("ROC CURVE")
	# plt.show()
	# print "yuyu"
	return TPR,FAR,ACC,fpr,tpr,threasholds





	# xx=[]
	# yy=[]
	# for t in np.arange(0,18,0.001):
	# 	result = []
	# 	for i in range(len(test_data)):
	# 		if(threashods[i] > t):
	# 			result.append(C1)
	# 		else:
	# 			result.append(C2)

	# 	a=0
	# 	b=0
	# 	c=0
	# 	d=0
	# 	for i in range(len(test_data)):
	# 		if(result[i] == C1 and test_labels[i]==C1):
	# 			a+=1
	# 		elif(result[i]==C1 and test_labels[i]==C2):
	# 			b+=1
	# 		elif result[i] == C2 and test_labels[i]==C1:
	# 			c+=1
	# 		elif result[i] == C2 and test_labels[i]==C2:
	# 			d+=1
	# 	far = c*1.0/(c+d)
	# 	tpr = a*1.0/(a+b)
	# 	xx.append(far)
	# 	yy.append(tpr)
	# plt.plot(yy,xx,"o",label = "Roc Curve")
	# plt.plot([0,1],[0,1])
	# plt.xlabel("TPR")
	# plt.ylabel("GAR")
	# plt.title("ROC CURVE")
	# plt.legend(loc='upper left')
	# plt.show()