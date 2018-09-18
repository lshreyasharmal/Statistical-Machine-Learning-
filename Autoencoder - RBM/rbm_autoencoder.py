from sklearn.neural_network import BernoulliRBM
from keras.datasets import mnist 
import numpy as np 
from keras.datasets import mnist 
import numpy as np
from keras import metrics 
import matplotlib.pyplot as plt 
from keras.utils import plot_model
from keras.models import Model 
from keras.layers import Input, Dense
from keras import regularizers

def load_data():
	(x_train,y_train),(x_test,y_test) = mnist.load_data()
	x_train = x_train.astype('float32')/255.0
	x_test = x_test.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
	print x_train.shape
	print x_test.shape
	return x_train,x_test
def autoencoder(x_train,x_test,func,encoding_dim,components,bias1,bias2):

	input_img = Input(shape=(784,))
	encoded = Dense(encoding_dim,activation=func,weights=[components,bias1])(input_img)
	decoded = Dense(784,activation='sigmoid',weights=[components.T,bias2])(encoded)
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
	np.save("relu_c",e3)
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
	plt.savefig("Question2c_relu")
	plt.show()




x_train,x_test = load_data()
rbm = BernoulliRBM(n_components = 256, n_iter = 10,learning_rate = 0.01,  verbose = True)
rbm.fit(x_train)
components = rbm.components_
bias1 = rbm.intercept_hidden_
bias2 = rbm.intercept_visible_
print components.shape
components = np.array(components)
bias1 = np.array(bias1)
bias2 = np.array(bias2)
np.save("rbm_weights",components)
np.save("rbm_bias1",bias1)
np.save("rbm_bias2",bias2)
# components = np.load("rbm_weights.npy")
# bias1 = np.load("rbm_bias1.npy")
# bias2 = np.load("rbm_bias2.npy")

encoding_dim = 256 
x_train,x_test = load_data()
decoded_images = autoencoder(x_train,x_test,'relu',encoding_dim,components.T,bias1,bias2)
visualize(x_test,decoded_images)
