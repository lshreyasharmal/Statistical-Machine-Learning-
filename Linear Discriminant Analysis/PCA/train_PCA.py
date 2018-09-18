from sklearn.decomposition import PCA
import numpy as np
def train_pca(train_data,test_data):
	# Compute a PCA 
	n_components = 10
	images = np.concatenate((train_data,test_data),axis=0)
	pp = PCA(n_components=n_components, whiten=True).fit(images)
	# print "ffff"
	# print pp
	# apply PCA transformation
	train_data_pca = pp.transform(train_data)
	test_data_pca = pp.transform(test_data)
	print "Training Done"
	return train_data_pca,test_data_pca