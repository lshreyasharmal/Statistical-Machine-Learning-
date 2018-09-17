# import main
# from mmain import *
import numpy as np

def func2(test, train, num_low, num_high, trainy, testy):
	print "Training..."

	#using k = 1 for laplace smoothing
	print "Calculating feature likelihoods..."
	#feature 1
	feature1 = []
	for i in range(1,3):
		x1 = 0
		x2 = 0
		for j in range(len(train)):
			if( train[j][0] == str(i) and trainy[j]=='1'):
				x1+=1
			elif( train[j][0]==str(i) and trainy[j]=='3'):
				x2+=1
		l = []
		l.append((x1*1.0+1)/(num_low+2))
		l.append((x2*1.0+1)/(num_high+2))
		feature1.append(l)

	#feature 2
	feature2 = []
	for i in range(1,26):
		x1 = 0
		x2 = 0
		for j in range(len(train)):
			if( train[j][1] == str(i) and trainy[j]=='1'):
				x1+=1
			elif( train[j][1]==str(i) and trainy[j]=='3'):
				x2+=1
		l = []
		l.append((x1*1.0+1)/(num_low+25))
		l.append((x2*1.0+1)/(num_high+25))
		feature2.append(l)

	#feature 3
	feature3 = []
	for i in range(1,27):
		x1 = 0
		x2 = 0
		for j in range(len(train)):
			if( train[j][2] == str(i) and trainy[j]=='1'):
				x1+=1
			elif( train[j][2]== str(i) and trainy[j]=='3'):
				x2+=1
		l = []
		l.append((x1*1.0+1)/(num_low+26))
		l.append((x2*1.0+1)/(num_high+26))
		feature3.append(l)

	#feature 4
	feature4 = []
	for i in range(1,3):
		x1 = 0
		x2 = 0
		for j in range(len(train)):
			# print train[j]
			if( train[j][3] == str(i) and trainy[j]=='1'):
				x1+=1
			elif( train[j][3]== str(i) and trainy[j]=='3'):
				x2+=1
		l = []
		l.append((x1*1.0+1)/(num_low+2))
		l.append((x2*1.0+1)/(num_high+2))
		feature4.append(l)

	#feature 5
	#converting continuous feature to discrete
	label_discrete = []
	for i in range(len(train)):
		temp = int(train[i][4]) 
		if(temp<= 10):
			label_discrete.append(0)
		elif(10<temp and temp<=20):
			label_discrete.append(1)
		elif 20<temp and temp<=30:
			label_discrete.append(2)
		elif 30<temp and temp<= 40:
			label_discrete.append(3)
		elif 40<temp and temp<=50:
			label_discrete.append(4)
		elif 50<temp and temp<60:
			label_discrete.append(5)
		else:
			label_discrete.append(6)
	#there are 7 classes 

	feature5 = []
	for i in range(7):
		x1 = 0
		x2 = 0
		for j in range(len(train)):
			if( label_discrete[j] == i and trainy[j]=='1'):
				x1+=1
			elif(label_discrete[j]== i and trainy[j]=='3'):
				x2+=1
		l = []
		l.append((x1*1.0+1)/(num_low+2))
		l.append((x2*1.0+1)/(num_high+2))
		feature5.append(l)
	
	return feature1, feature2, feature3, feature4, feature5, test, testy, train, trainy, num_high, num_low
