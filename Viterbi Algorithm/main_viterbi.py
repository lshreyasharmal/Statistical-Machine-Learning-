from viterbi import viterbi_
alph_num = {}
for i in range(1,27):
	alph_num[i] = chr(ord('a')+i-1)

hiddenStates = [i for i in range(1,27)]
obsStates = [i for i in range(2)]
transprobmatrix = []
with open("transitionProbMatrix.txt") as f:
	for line in f:
		temp = line.split(",")
		temp = [float(i) for i in temp]
		transprobmatrix.append(temp)
initstatesdis = []
with open("initialStateDistribution.txt") as f:
	for line in f:
		initstatesdis.append(float(line))
obsprobmatrix = []
with open("observationProbMatrix.txt") as f:
	for line in f:
		temp = line.split(",")
		temp = [float(i) for i in temp]
		obsprobmatrix.append(temp)
obsart = []
with open("observations_art.txt") as f:
	for line in f:
		line = line.split(",")
		obsart = [int(i) for i in line]
obstest = []
with open("observations_test.txt") as f:
	for line in f:
		line = line.split(",")
		if(line[len(line)-1]=='\n'):
			del line[len(line)-1]
		obstest = [int(i) for i in line]
# print transprobmatrix
# print initstatesdis
# print obsprobmatrix
# print obsart
# print obstest


T = len(obsart)
output = viterbi_(hiddenStates,obsStates,transprobmatrix,initstatesdis,obsprobmatrix,obsart,T)
string = ""
for o in output:
	string+=alph_num[o]
print string




T = len(obstest)
output = viterbi_(hiddenStates,obsStates,transprobmatrix,initstatesdis,obsprobmatrix,obstest,T)

# print output

string = ""
for o in output:
	string+=alph_num[o]
print string

f = open("out_seq.txt","w+")
for s in string:
	f.write(s+"\n")
f.close()
