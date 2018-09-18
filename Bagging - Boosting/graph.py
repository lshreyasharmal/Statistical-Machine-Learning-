import matplotlib.pyplot as plt

acc = [0.2720,0.304,0.3105]
x = [1,2,3]
plt.plot(x,acc,'ro')
plt.title("Accuracy vs Models")
plt.xlabel("Models")
plt.ylabel("Accuracies")
plt.xticks(x,['New Classifier','Bagging with 5 estimators','Bagging with 15 estimators'])
plt.show()
