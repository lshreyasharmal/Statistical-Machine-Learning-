from sklearn.naive_bayes import GaussianNB
import numpy as np
import numpy.linalg as LA

def test_pca(train_data,train_data_labels,test_data,test_data_labels):
	# print "ijo"
	clf = GaussianNB()
	clf.fit(train_data, train_data_labels)
	result = clf.predict(test_data)
	acc = 0
	for i in range(len(result)):
		if result[i]==test_data_labels[i]:
			acc+=1
	acc = acc*100.0/len(result)
	# return (acc*100.0/len(result))
	print acc

	print "ROC:-"
	pairs = []
	class_labels = []

	TPRS = []
	FPRS = []

	for i in range(len(test_data)-1):
		for j in range(i+1,len(test_data)):
			f=[]
			f.append(test_data[i])
			f.append(test_data[j])
			g = []
			g.append(test_data_labels[i])
			g.append(test_data_labels[j])
			g = np.array(g)
			f=np.array(f)
			pairs.append(f)
			class_labels.append(g)
	pairs = np.array(pairs)	
	class_labels = np.array(class_labels)
	Y = []
	for p in range(len(pairs)):
		if class_labels[p][0]==class_labels[p][1]:
			Y.append(0)
		else:
			Y.append(1)
	Y = np.array(Y)
	m = -10
	for p in range(len(pairs)):
		a = pairs[p][0]
		b = pairs[p][1]
		d =  LA.norm(a-b)
		if(d>m):
			m=d
	result=[]
	pairs_dis =[]

	for p in range(len(pairs)):
		a = pairs[p][0]
		b = pairs[p][1]
		d =  LA.norm(a-b)
		d=d*1.0/m
		pairs_dis.append(d)
		# print d
		
	for t in np.arange(0.1,1,0.01):
		result=[]
		for p in range(len(pairs)):
			if pairs_dis[p]>=t:
				result.append(1)
			else:
				result.append(0)
		a=0
		b=0
		c=0
		d=0
		# print result
		for r in range(len(result)):
			if result[r]==0 and Y[r]==0:
				a+=1
			elif result[r]==1 and Y[r]==0:
				b+=1
			elif result[r]==0 and Y[r]==1:
				c+=1
			elif result[r]==1 and Y[r]==1:
				d+=1
			# print result[r]
			# print Y[r]
		# print a
		# print b
		# print c
		# print d
		tpr = a*1.0/(a+b)
		fpr = c*1.0/(c+d)
		TPRS.append(tpr)
		FPRS.append(fpr)
	print len(TPRS)
	return acc,TPRS,FPRS
