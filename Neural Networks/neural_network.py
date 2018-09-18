import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.interpolate import interp1d
from numpy import linalg as LA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
def relu(val):
    return np.maximum(0,val)
def derivative_relu(x):
	# print x
	x[x<=0] = 0 
	x[x>0] = 1
	return x
def softmax(z):
	numerator = np.exp(z)
	return numerator/np.sum(numerator,axis=1,keepdims=True)
def prop_forward(trainX):
	z1 = np.add(trainX.dot(wji),bj)
	hidden_layer_output = relu(z1)
	z2 = np.dot(hidden_layer_output,wkj)+bk
	output = softmax(z2)
	return output,hidden_layer_output
def gradients(output):
	dscores = output 
	dscores[range(n),trainY] -= 1
	dscores /= n
	return dscores
def back_prop(trainX,hidden_layer_output,dscores):
	dw2 = np.dot(hidden_layer_output.T,dscores)
	db2 = np.sum(dscores,axis=0,keepdims=True)
	dhidden = np.dot(dscores,wkj.T)
	dhidden[hidden_layer_output<=0] = 0
	dw = np.dot(trainX.T,dhidden)
	db = np.sum(dhidden,axis=0,keepdims=True)
	dw2 += reg*wkj
	dw += reg*wji
	return dw,db,dw2,db2


workclass = {'Private':1,'Self-emp-not-inc':2,'Self-emp-inc':3,'Federal-gov':4,'Local-gov':5,'State-gov':6,'Without-pay':7,'Never-worked':8}
education = {'Bachelors':1, 'Some-college':2, '11th':3, 'HS-grad':4, 'Prof-school':5, 'Assoc-acdm':6, 'Assoc-voc':7, '9th':8, '7th-8th':9, '12th':10, 'Masters':11, '1st-4th':12, '10th':13, 'Doctorate':14, '5th-6th':15, 'Preschool':16}
maritinal_status = {"Married-civ-spouse":1, "Divorced":2, "Never-married":3, "Separated":4, "Widowed":5, "Married-spouse-absent":6, "Married-AF-spouse":7}
occupation = {"Tech-support":1, "Craft-repair":2, "Other-service":3, "Sales":4, "Exec-managerial":5, "Prof-specialty":6, "Handlers-cleaners":7, "Machine-op-inspct":8, "Adm-clerical":9, "Farming-fishing":10, "Transport-moving":11, "Priv-house-serv":12, "Protective-serv":13, "Armed-Forces":14}
relationship =  {"Wife":1, "Own-child":2, "Husband":3, "Not-in-family":4, "Other-relative":5, "Unmarried":6}
race = {"White":1, "Asian-Pac-Islander":2, "Amer-Indian-Eskimo":3, "Other":4, "Black":5}
sex = {'Female':1, 'Male':2}
native_country = {"United-States":1, "Cambodia":2, "England":3, "Puerto-Rico":4, "Canada":5, "Germany":6, "Outlying-US(Guam-USVI-etc)":7, "India":8, "Japan":9, "Greece":10, "South":11, "China":12, 'Cuba':13, 'Iran':14, 'Honduras':15, 'Philippines':16, 'Italy':17, 'Poland':18, 'Jamaica':19, 'Vietnam':20, 'Mexico':21, 'Portugal':22, 'Ireland':23, 'France':24, 'Dominican-Republic':25, 'Laos':26, 'Ecuador':27, 'Taiwan':28, 'Haiti':29, 'Columbia':30, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40}
class0 = []
class1 = []
with open("dataset",'r') as file:
	for line in file:

		line = line.replace(" ", "")
		line = line.split(",")
		# print line 
		if "?" in line:
			continue
		# dataset.append(line[0:len(line)-1])
		for i in range(len(line)-1):
			# print i
			if i == 1:
				line[i] = float(workclass[line[i]])
			elif i == 3:
				line[i] = float(education[line[i]])
			elif i == 5:
				line[i] = float(maritinal_status[line[i]])
			elif i == 6:
				line[i] = float(occupation[line[i]])
			elif i == 7:
				line[i] = float(relationship[line[i]])
			elif i == 8:
				line[i] = float(race[line[i]])
			elif i == 9:
				line[i] = float(sex[line[i]])
			elif i == 13:
				line[i] = float(native_country[line[i]])
			else:
				line[i] = float(line[i])
		# print line[len(line)-1]
		if line[len(line)-1].strip() == "<=50K":
			class0.append(line[0:len(line)-1])
		elif line[len(line)-1].strip() == ">50K":
			class1.append(line[0:len(line)-1])

class0 = np.array(class0)
class1 = np.array(class1)

trainX = []
trainY = []
testX = []
testY = []

class0_indices = random.sample(range(len(class0)), int(0.5*len(class0)))
class1_indices = random.sample(range(len(class1)), int(0.5*len(class1)))

for i in class0_indices:
	trainX.append(class0[i])
	trainY.append(0)
for i in class1_indices:
	trainX.append(class1[i])
	trainY.append(1)

trainX = np.array(trainX)
trainY = np.array(trainY)
print "length of class 0 : " + str(len(class0))
print "length of class 1 : "+ str(len(class1))
s = np.arange(trainX.shape[0])
np.random.shuffle(s)

trainX = trainX[s]
trainY = trainY[s]

for i in range(len(class0)):
	if i not in class0_indices:
		testX.append(class0[i])
		testY.append(0)
for i in range(len(class1)):
	if i not in class1_indices:
		testX.append(class1[i])
		testY.append(1)

testX = np.array(testX)
testY = np.array(testY)
s = np.arange(testX.shape[0])
np.random.shuffle(s)

testX = testX[s]
testY = testY[s]

scaler = StandardScaler()
scaler.fit(trainX)

trainX = scaler.transform(trainX)
testX = scaler.transform(testX)


print "length of training data = " + str(len(trainX))
print "length of testing data = " + str(len(testX))

ephocs = 1000
learning_rate = 1e-1
wji = np.random.randn(14, 3) * 0.001
bj = np.zeros((1,3))
wkj = np.random.randn(3,2) * 0.001
bk = np.zeros((1,2))
error = np.array([])
prev_Jw = 0
reg = 1e-3
n = trainX.shape[0]
# print n
print "training neural network ..."
print 
for i in range(ephocs):
	output,hidden_layer_output = prop_forward(trainX)
	dscores = gradients(output)
	dw,db,dw2,db2 = back_prop(trainX,hidden_layer_output,dscores)

	wji += -learning_rate*dw
	bj += (-learning_rate*db)
	wkj += -learning_rate*dw2
	bk += (-learning_rate*db2)

output,hidden_layer_output = prop_forward(trainX)
predicted_class = np.argmax(output, axis=1)
print 'training accuracy: %.6f' % (np.mean(predicted_class == trainY))

output,hidden_layer_output = prop_forward(testX)
predicted_class = np.argmax(output, axis=1)
print 'testing accuracy: %.6f' % (np.mean(predicted_class == testY))

print 
print "CONFUSION MATRIX "
tn, fp, fn, tp = confusion_matrix(testY,predicted_class).ravel()
total = tn+fp+fn+tp
print "	  | predicted:1 | predicted:0 |"
# print "-----------------------------------------"
print "|actual:1 " + "|tn =  %.3f  |fp =   %.3f |" %((tn*1.0/total),fp*1.0/total)
print "|actual:0 " + "|fn =  %.3f  |tp =   %.3f |" %(fn*1.0/total,tp*1.0/total)


fpr, tpr, threasholds = metrics.roc_curve(testY,output[:,1],pos_label=1)
roc_auc = metrics.auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='r',label='ROC curve (area=%0.2f'%roc_auc)
plt.plot([0,1],[0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc ='lower right')
plt.title("ROC CURVE")
plt.show()


# print "Th 	tpr 	fpr"
# for i in range(len(fpr)):
# 	print "%.6f	%.6f	%.6f" %(threasholds[i],1-tpr[i],fpr[i])
# 	if 1-tpr[i] == fpr[i]:
# 		print threasholds[i]
# 		break

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

print "EER VALUE : %.4f" %(eer)
