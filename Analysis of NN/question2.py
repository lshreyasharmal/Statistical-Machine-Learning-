import csv
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

train_labels = []
test_labels = []
train_data = []
test_data = []
with open("emnist-balanced-train.csv",'r') as trainfile:
	reader = csv.reader(trainfile)
	for row in reader:	
		train_data.append([float(x) for x in row[1:len(row)]])
		train_labels.append(int(row[0]))

with open("emnist-balanced-test.csv",'r') as testfile:
	reader = csv.reader(testfile)
	for row in reader:	
		test_data.append([float(x) for x in row[1:len(row)]])
		test_labels.append(int(row[0]))
print "splitting"


# scaler = StandardScaler()
# scaler.fit(test_data)

# train_data = scaler.transform(train_data)
# test_data = scaler.transform(test_data)
# print len(train_labels)
# print len(test_labels)
# train_data=train_data[0:1000]
# test_data=test_data[0:200]
# train_labels=train_labels[0:1000]
# test_labels=test_labels[0:200]
lr = [0.1]
# lr = [0.2,0.1,0.001]
epochs = [100]
acc = []
for l in lr:
	for e in epochs:
		# print "constructing model..."
		# mlp = MLPClassifier(hidden_layer_sizes=(64,128,256),activation = 'identity',learning_rate_init=l,max_iter=e)
		# print "training..."
		# mlp.fit(train_data,train_labels)
		# print "testing..."
		# predictions = mlp.predict(test_data)

		# correct_classified = 0
		# for i in range(len(predictions)):
		# 	if predictions[i] == test_labels[i]:
		# 		correct_classified+=1
		# accuracy = float(correct_classified*100)/len(predictions)
		# print accuracy
		# acc.append(accuracy)
		# print "constructing model..."
		# mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'identity',learning_rate_init=l,max_iter=e)
		# print "training..."
		# mlp.fit(train_data,train_labels)
		# print "testing..."
		# predictions = mlp.predict(test_data)

		# correct_classified = 0
		# for i in range(len(predictions)):
		# 	if predictions[i] == test_labels[i]:
		# 		correct_classified+=1
		# accuracy = float(correct_classified*100)/len(predictions)
		# print accuracy
		# acc.append(accuracy)
		# print "constructing model..."
		# mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'relu',learning_rate_init=l,max_iter=e)
		# print "training..."
		# mlp.fit(train_data,train_labels)
		# print "testing..."
		# predictions = mlp.predict(test_data)

		# correct_classified = 0
		# for i in range(len(predictions)):
		# 	if predictions[i] == test_labels[i]:
		# 		correct_classified+=1
		# accuracy = float(correct_classified*100)/len(predictions)
		# print accuracy
		# acc.append(accuracy)
		# print "constructing model..."
		# mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'logistic',learning_rate_init=l,max_iter=e)
		# print "training..."
		# mlp.fit(train_data,train_labels)
		# print "testing..."
		# predictions = mlp.predict(test_data)

		# correct_classified = 0
		# for i in range(len(predictions)):
		# 	if predictions[i] == test_labels[i]:
		# 		correct_classified+=1
		# accuracy = float(correct_classified*100)/len(predictions)
		# print accuracy
		# acc.append(accuracy)

		# print "constructing model..."
		# mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'identity',learning_rate_init=l,max_iter=e)
		# print "training..."
		# mlp.fit(train_data,train_labels)
		# print "testing..."
		# predictions = mlp.predict(test_data)

		# correct_classified = 0
		# for i in range(len(predictions)):
		# 	if predictions[i] == test_labels[i]:
		# 		correct_classified+=1
		# accuracy = float(correct_classified*100)/len(predictions)
		# print accuracy
		# acc.append(accuracy)
		
		# print "constructing model..."
		# mlp = MLPClassifier(hidden_layer_sizes=(128,256,64),activation = 'identity',learning_rate_init=l,max_iter=e)
		# print "training..."
		# mlp.fit(train_data,train_labels)
		# print "testing..."
		# predictions = mlp.predict(test_data)

		# correct_classified = 0
		# for i in range(len(predictions)):
		# 	if predictions[i] == test_labels[i]:
		# 		correct_classified+=1
		# accuracy = float(correct_classified*100)/len(predictions)
		# print accuracy
		# acc.append(accuracy)

		print "constructing model..."
		mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'identity',learning_rate_init=l,max_iter=e,alpha=0.0001)
		print "training..."
		mlp.fit(train_data,train_labels)
		print "testing..."
		predictions = mlp.predict(test_data)
		correct_classified = 0
		for i in range(len(predictions)):
			if predictions[i] == test_labels[i]:
				correct_classified+=1
		accuracy = float(correct_classified*100)/len(predictions)
		print accuracy
		acc.append(accuracy)
		print "constructing model..."
		mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'identity',learning_rate_init=l,max_iter=e,alpha=0.001)
		print "training..."
		mlp.fit(train_data,train_labels)
		print "testing..."
		predictions = mlp.predict(test_data)
		correct_classified = 0
		for i in range(len(predictions)):
			if predictions[i] == test_labels[i]:
				correct_classified+=1
		accuracy = float(correct_classified*100)/len(predictions)
		print accuracy
		acc.append(accuracy)
		print "constructing model..."
		mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'identity',learning_rate_init=l,max_iter=e,alpha=0.01)
		print "training..."
		mlp.fit(train_data,train_labels)
		print "testing..."
		predictions = mlp.predict(test_data)
		correct_classified = 0
		for i in range(len(predictions)):
			if predictions[i] == test_labels[i]:
				correct_classified+=1
		accuracy = float(correct_classified*100)/len(predictions)
		print accuracy
		acc.append(accuracy)
		print "constructing model..."
		mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),activation = 'logistic',learning_rate_init=l,max_iter=e,alpha=0.1)
		print "training..."
		mlp.fit(train_data,train_labels)
		print "testing..."
		predictions = mlp.predict(test_data)
		correct_classified = 0
		for i in range(len(predictions)):
			if predictions[i] == test_labels[i]:
				correct_classified+=1
		accuracy = float(correct_classified*100)/len(predictions)
		print accuracy
		acc.append(accuracy)

print acc


plt.plot([1,2,3,4],acc)
plt.ylabel("Accuracy")
plt.xlabel("alpha values")
plt.title("Accuracy vs alpha values")
plt.xticks([1,2,3,4],[0.0001,0.001,0.01,0.1])
plt.yticks(np.arange(30,101,10))
plt.show()


