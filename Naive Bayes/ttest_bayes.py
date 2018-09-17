import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
def testt(feature1, feature2, feature3, feature4, feature5, test, testy, train, trainy, num_high, num_low):


	likelihood_low_arr = []
	likelihood_high_arr = []
	output=[]

	p1 = num_low*1.0/(num_low+num_high)
	p2 = num_high*1.0/(num_high+num_low)

	print "The Prior Probabilities of class low and high are : %f , %f"%(p1,p2)

	threashods = []
	print "Testing data..."
	for i in range(len(test)):
		temp = int(test[i][4]) 
		g = 0
		if(10<temp and temp<=20):
			g=1
		elif 20<temp and temp<=30:
			g=2
		elif 30<temp and temp<= 40:
			g=3
		elif 40<temp and temp<=50:
			g=4
		elif 50<temp and temp<= 60:
			g=5
		else:
			g=6

		#Calculating likelihoods
		likelihood_low = feature1[int(test[i][0])-1][0]*feature2[int(test[i][1])-1][0]*feature3[int(test[i][2])-1][0]*feature4[int(test[i][3])-1][0]*feature5[g][0]
		likelihood_high = feature1[int(test[i][0])-1][1]*feature2[int(test[i][1])-1][1]*feature3[int(test[i][2])-1][1]*feature4[int(test[i][3])-1][1]*feature5[g][1]



		likelihood_low_arr.append(likelihood_low)
		likelihood_high_arr.append(likelihood_high)


		# print str(likelihood_high*1.0/(likelihood_high+likelihood_low)) + " " + str(likelihood_low*1.0/(likelihood_low+likelihood_high))
		if((likelihood_low*p1)/(likelihood_high*p2)>1):
			output.append('1')
		else:
			output.append('3')
		threashods.append((likelihood_low*p1)/(likelihood_high*p2))
	total = num_low+num_high
	acc = 0

	a=0
	b=0
	c=0
	d=0
	for i in range(len(test)):
		if(output[i] == '1' and testy[i]=='1'):
			a+=1
		elif(output[i]=='3' and testy[i]=='1'):
			b+=1
		elif output[i] == '1' and testy[i]=='3':
			c+=1
		elif output[i] == '3' and testy[i]=='3':
			d+=1
	far = c*1.0/(c+d)
	tpr = a*1.0/(a+b)
	print "False Acceptance Rate = %f"%(far)
	print "True Positive Rate = %f"%(tpr)

	acc = (a+d)*1.0/(a+b+c+d)
	print "ACCURACY = %f"%(acc)
	print "Training Samples belonging to class low = %d , class high = %d"%(num_high,num_low)

	print "Confusion Matrix"
	print "-------------------------------------------"
	print "Actual/Predicted	Low 	High 	Total"
	print "Low 			" + str(a)+ "	"+str(b) + "	" +str(a+b)
	print "High 			" + str(c) + "	"+str(d)+"	"+ str(c+d)
	print "Total			"+str(a+c) + "	" + str(d+b)
	print "-------------------------------------------"
	print "Length of Training data = %d, Length of Testing data = %d"%(int(len(train)),int(len(test)))


	print "ROC CURVE"
	xx = []
	yy = []

	# print threashods
	for t in np.arange(0,18,0.001):
		# print t
		output = []
		for i in range(len(test)):
			if(threashods[i] > t):
				output.append('1')
			else:
				output.append('3')

		a=0
		b=0
		c=0
		d=0
		for i in range(len(test)):
			if(output[i] == '1' and testy[i]=='1'):
				a+=1
			elif(output[i]=='3' and testy[i]=='1'):
				b+=1
			elif output[i] == '1' and testy[i]=='3':
				c+=1
			elif output[i] == '3' and testy[i]=='3':
				d+=1
		far = c*1.0/(c+d)
		tpr = a*1.0/(a+b)
		xx.append(far)
		yy.append(tpr)
	plt.plot(xx,yy,"o",label = "Roc Curve")
	plt.plot([0,1],[0,1])
	plt.xlabel("False Acceptance Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC CURVE")
	plt.legend(loc='upper left')
	plt.show()
	return num_high, num_low, acc, likelihood_low_arr, likelihood_high_arr


	


