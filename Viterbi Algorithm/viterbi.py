import math
import numpy as np


def viterbi_(hiddenStates,obsStates,transprobmatrix,initstatesdis,obsprobmatrix,obsart,T):
	V = []
	Vk1 = []
	for k in hiddenStates:
		Vk1.append(math.log(obsprobmatrix[k-1][obsart[0]])+math.log(initstatesdis[k-1]))
	Vk1 = np.array(Vk1)
	V.append(Vk1)
	
	for t in range(1,T):
		temp = []
		Vkt = []
		for k in hiddenStates:
			temp = [math.log(transprobmatrix[i-1][k-1])+V[t-1][i-1] for i in hiddenStates]
			temp = np.array(temp)
			Vkt.append(math.log(obsprobmatrix[k-1][obsart[t]])+max(temp))
		Vkt = np.array(Vkt)
		V.append(Vkt)
	path = []
	V = np.array(V)
	path.append(np.argmax(V[T-1])+1)
	for t in range(T-1,0,-1):
		temp = np.array([math.log(transprobmatrix[i-1][path[T-1-t]-1])+V[t-1][i-1] for i in hiddenStates])
		path.append(np.argmax(temp)+1)
	path = np.array(path)
	path = path[::-1]
	return path