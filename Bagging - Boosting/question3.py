import cPickle
import numpy as np
from skimage.io import imsave
import cv2
import hep_ml.nnet
from hep_ml.nnet import MLPMultiClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
def save_as_image(img_flat):
	img_R = img_flat[0:1024].reshape((32, 32))
	img_G = img_flat[1024:2048].reshape((32, 32))
	img_B = img_flat[2048:3072].reshape((32, 32))
	img = np.dstack((img_R, img_G, img_B))
	return img
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict
trainX = []
trainY = []
for k in range(1,2):
	# print "j"
	folder = 'cifar-10-batches-py/data_batch_'+str(k)
	dic = unpickle(folder)
	trainX.extend(dic['data'])
	trainY.extend(dic['labels'])
trainX = np.array(trainX)
trainY = np.array(trainY)

temp = []
i = 0
for t in trainX:
	if i == 1000:
		break
	i+=1
	img = save_as_image(t)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = img.reshape(1024)
	temp.append(img)
temp = np.array(temp)
trainX = temp
trainY = trainY[0:1000]
testX = []
testY = []
folder = 'cifar-10-batches-py/test_batch'
dic = unpickle(folder)
testX.extend(dic['data'])
testY.extend(dic['labels'])
testX = np.array(testX)
testY = np.array(testY)
# print type(trainY[0])
temp = []
i=0
for t in testX:
	if i == 100:
		break
	i+=1
	img = save_as_image(t)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = img.reshape(1024)
	temp.append(img)
temp = np.array(temp)
testX = temp
testY = testY[0:100]
print trainY.shape

trainX = trainX.astype(float)
testX = testX.astype(float)
scaler = StandardScaler()
scaler.fit(trainX)

trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

# print "Whithout anything"
classifier = MLPClassifier(hidden_layer_sizes=(50))
classifier = MLPClassifier(hidden_layer_sizes=(400,150)) 
classifier = MLPMultiClassifier(layers=(50,),trainer='adadelta',epochs=10)
classifier = MLPMultiClassifier(layers=(400,150,),trainer='adadelta',epochs=10)
# print "With boosting"
classifier = AdaBoostClassifier(base_estimator=classifier,learning_rate=0.1)
classifier = AdaBoostClassifier(base_estimator=classifier,learning_rate=0.1,n_estimators=5)
classifier = AdaBoostClassifier(base_estimator=classifier,learning_rate=0.1,n_estimators=15)
# print "-0.1-"
classifier = BaggingClassifier(base_estimator=classifier)
classifier = BaggingClassifier(base_estimator=classifier,n_estimators=5)
classifier = BaggingClassifier(base_estimator=classifier,n_estimators=15)
classifier.fit(trainX, trainY)
# print "15"
predictions = classifier.predict(testX)
print predictions

print 'testing accuracy: %.6f' % (np.mean(predictions == testY))
