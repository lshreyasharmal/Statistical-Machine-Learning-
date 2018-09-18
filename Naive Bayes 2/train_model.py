import numpy as np
def train_(train_data,train_labels,C1,C2):
	#normalise data
	# C1 = 0
	# C2 = 1
	C = [C1,C2]
	for i in range(len(train_data)):
		for x in range(len(train_data[i])):
			train_data[i][x]/=255.0
	for i in range(len(train_data)):
		for x in range(len(train_data[i])):
			if train_data[i][x]>0.5:
				train_data[i][x]=1
			else:
				train_data[i][x]=0
	n1 = 0
	n2 = 0
	for x in train_labels:
		if x == C1:
			n1+=1;
		else:
			n2+=1
	p1 = n1*1.0/(n1+n2)
	p2 = 1-p1

	model_high = []
	model_low = []
	for i in range(784):
		for k in C:
			x1=0	
			x2=0
			for j in range(len(train_data)):
				if(train_data[j][i]==0 and train_labels[j]==k):
					x1+=1
				elif train_data[j][i]==1 and train_labels[j]==k:
					x2+=1

			# print l
			if(k==C1):
				l = []
				l.append((x1*1.0+1)/(n1+2))
				l.append((x2*1.0+1)/(n1+2))
				model_low.append(l)
			else:
				l = []
				l.append((x1*1.0+1)/(n2+2))
				l.append((x2*1.0+1)/(n2+2))
				model_high.append(l)


	return model_low, model_high,p1,p2