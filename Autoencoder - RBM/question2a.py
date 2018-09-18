from keras.datasets import mnist 
import numpy as np
from keras import metrics 
import matplotlib.pyplot as plt 
from keras.models import Model 
from keras.layers import Input, Dense
from keras import regularizers
# from tempfile import TemporaryFile
def load_data():
	(x_train,y_train),(x_test,y_test) = mnist.load_data()
	x_train = x_train.astype('float32')/255.0
	x_test = x_test.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
	print x_train.shape
	print x_test.shape
	return x_train,x_test

def autoencoder(x_train,x_test,func,encoding_dim):

	input_img = Input(shape=(784,))
	encoded = Dense(encoding_dim,activation=func)(input_img)
	decoded = Dense(784,activation='sigmoid')(encoded)
	autoencoder = Model(input_img,decoded)
	encoder = Model(input_img,encoded)
	encoded_input = Input(shape=(encoding_dim,))
	decoded_layer = autoencoder.layers[-1]
	decoder = Model(encoded_input,decoded_layer(encoded_input))
	autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=[metrics.mse,'acc'])
	autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,validation_data=(x_test,x_test))
	encoded_images = encoder.predict(x_test)
	decoded_images = decoder.predict(encoded_images)
	e3 = np.array(autoencoder.get_weights())
	np.save("sigmoid",e3)
	return decoded_images

def visualize(x_test,decoded_images):
	n = 15
	plt.figure(figsize=(20,4))
	for i in range(n):
		ax = plt.subplot(2,n,i+1)
		plt.imshow(x_test[i].reshape(28,28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax = plt.subplot(2,n,i+1+n)
		plt.imshow(decoded_images[i].reshape(28,28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.savefig("Question2a_sigmoid")
	plt.show()

encoding_dim = 256 
x_train,x_test = load_data()
decoded_images = autoencoder(x_train,x_test,'sigmoid',encoding_dim)
visualize(x_test,decoded_images)
