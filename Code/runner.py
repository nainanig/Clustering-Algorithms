import numpy as np
import subprocess
import sys
import pandas as pd
from sklearn.decomposition import PCA

converged=False
k=10
maxIterations=100
centroidDataFile='data.txt'
inputFile='iyer.txt'

initial = np.loadtxt(inputFile, delimiter="\t",dtype='str')
#idx = np.random.randint(initial.shape[0], size=5)
idx=[309,97,501,278,429,266,142,286,273,272] # SUBTRACT 1

print"Initial Centers: " ,idx
idx[:] = [x - 1 for x in idx]


matrix=initial[:,2:]

initialCentroid=matrix[idx,:]
initialCentroid=initialCentroid.tolist()

fileName = open(centroidDataFile, 'w')# file 
index=1
for centroid in initialCentroid:
	st='\t'.join((map(str,centroid)))

	fileName.write("%s\t%s\n" % (index,st))
	index=index+1
fileName.close()


iterations=0

def checkConversion(new, old):
	if np.array_equal(new,old):
			return True
	return False

def metric(matrixa, matrixb):
	m11=0; m10=0; m01=0; m00=0;
	for i in range(len(matrixa)):
		for j in range(len(matrixb)):
			if matrixa[i][j]==matrixb[i][j]==1:
				m11=m11+1
			elif matrixa[i][j]==matrixb[i][j]==0:
				m00=m00+1
			elif matrixa[i][j]==1 and matrixb[i][j]==0:
				m10=m10+1
			elif matrixa[i][j]==0 and matrixb[i][j]==1:
				m01=m01+1
	rand = float(m11 +m00)/(m00+m11+m01+m10)
	jak=float(m11)/(m11+m10+m01)
	print"rand :" ,rand
	print "jaccard :" ,jak


def eucli_dist(matrix1,matrix2):
	
	distance = np.linalg.norm(matrix1 - matrix2)
	return distance


while(converged!=True and iterations < maxIterations):
	x=subprocess.Popen('rm -r output/', shell=True)
	x.wait()
	p=subprocess.Popen('/usr/local/hadoop/bin/hadoop jar hadoop-streaming-2.7.4.jar -file mapper.py -file reducer.py -file data.txt -mapper mapper.py -reducer reducer.py -input iyer.txt -output output/',shell=True) 
	p.wait()
	reducerOutput = np.loadtxt('/home/hp/Desktop/hduser/output/part-00000', delimiter="\t")
	centroidList=[]
	fileName = open('data.txt', 'w')
	for output in reducerOutput:
		#print (output)
		#temp=output.split("\t")

		t=output[1:]
		centroidList.append(t)
		st='\t'.join((map(str,t)))

		fileName.write("%s\t%s\n" % (int(output[0]),st))
	fileName.close()
	centroidList=np.array(centroidList, dtype=float)


	if(checkConversion(centroidList,initialCentroid)):
		converged=True
	
	initialCentroid=centroidList
	iterations=iterations+1


print(iterations)

result = []
for i in matrix:
	i=i.astype(np.float)
	closest = sys.maxint;pos=0;c=0;
	for j in initialCentroid:
		j=j.astype(np.float)
		#k=j[2:]
		diff=eucli_dist(i,j)
		#print(diff)
		if diff<closest:

			#print("Inside if")
			pos=c
			#print("diff", diff)
			closest=diff;

		c=c+1
	result.append(pos)

t = (matrix.shape[0],matrix.shape[0])
ground_matrix= np.zeros(t)
real_matrix= np.zeros(t)

for i in range(0,len(ground_matrix)): 
	for j in range(0,len(ground_matrix)):
		if(initial[i][1]==initial[j][1]):
			ground_matrix[i][j]=1
		else:
			ground_matrix[i][j]=0

for i in range(0,len(real_matrix)): 
	for j in range(0,len(real_matrix)):
		if(result[i]==result[j]):
			real_matrix[i][j]=1
		else:
			real_matrix[i][j]=0

def pca2(data,pc_count=None):
	return PCA(n_components=2).fit_transform(data)

metric(ground_matrix,real_matrix)
pca_dataset=pca2(initial[:,2:])
#real_labels=real_matrix[:,0]
#real_labels=np.reshape(real_labels,(real_labels.shape[0],1))
#print(real_labels)
#print(pca_dataset.shape)
length=len(result)
result=np.array(result)
result=np.reshape(result,(length,1))


#print(result)
forCSV=np.concatenate((pca_dataset,result),axis=1)
#print(forCSV)
df_kmeans=pd.DataFrame(forCSV)
df_kmeans.to_csv("out_mr.csv")
#print(pca_dataset)