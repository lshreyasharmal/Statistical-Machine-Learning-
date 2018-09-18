import numpy as np
import os
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import misc
import matplotlib.pyplot as plt
from pca import pca

def plot_gallery(images, h, w,r,c):
	# print images.shape
	plt.figure()
	for i in range(len(images)):
		plt.subplot(r,c,i+1)
		plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
		plt.title("eigenvalue "+ str(i+1))
		plt.xticks(())
		plt.yticks(())
	plt.show()

images = []
for i in range(1,12):
	folder = "./Face_data/"+str(i)+"/"
	temp = []
	for face in os.listdir(folder):
		if("yale" in face and "info" not in face):
			f = misc.imread(folder+face)
			# misc.imshow(f)
			temp.append(f)
			images.append(f)
images = np.array(images)

reshaped_images = []
for i in images:
	x = np.resize(i,(192,168))
	f = x.shape
	x = x.reshape(f[0]*f[1])
	reshaped_images.append(x)
reshaped_images = np.array(reshaped_images)
thresholds = [0.9,0.95,0.99]

for t in thresholds:
	print "threshold = "+str(t)
	print reshaped_images.shape	
	new_eigenvectors = pca(t,reshaped_images)
	new_faces = np.dot(new_eigenvectors,reshaped_images.T)
	new_faces = new_faces.T
	# print new_faces.shape
	h = 192
	w = 168
	r = new_eigenvectors.shape[0]
	rows = int(math.ceil(math.sqrt(r)))
	cols = int(math.ceil(r*1.0/rows))
	print rows 
	print cols
	plot_gallery(new_eigenvectors, h, w,rows,cols)


