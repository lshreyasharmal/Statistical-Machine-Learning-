from sklearn.neural_network import MLPClassifier
import numpy as np 
from keras.datasets import mnist
def load_data():
	(x_train,y_train),(x_test,y_test) = mnist.load_data()
	x_train = x_train.astype('float32')/255.0
	x_test = x_test.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
	# print x_train.shape
	# print x_test.shape
	y_test = np.array(y_test)
	y_train=np.array(y_train)
	return x_train,x_test,y_train,y_test


x_train,x_test,y_train,y_test = load_data()

temp = np.load("relu_c.npy")
weights = temp[0]
bias = temp[1]
print weights.shape
print bias.shape

print x_train.shape


x_train = np.matmul(x_train,weights) + bias
x_test = np.matmul(x_test,weights) + bias

# classifier = MLPClassifier(hidden_layer_sizes = 128, max_iter = 500, activation='logistic',solver='adam',learning_rate_init=0.01,early_stopping=True)
classifier = MLPClassifier(hidden_layer_sizes = 128, max_iter = 500, activation='logistic',solver='adam',learning_rate_init=0.01,early_stopping=True)
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)
print predictions

print 'testing accuracy: %.6f' % (np.mean(predictions == y_test))

#for i in range(len(y_test)):
#	if y_test[i]!=predictions[i]:
#		print str(y_test[i]) + " " + str(predictions[i])
