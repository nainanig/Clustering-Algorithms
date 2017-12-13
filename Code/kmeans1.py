from __future__ import division
import numpy as np;
import pandas as pd;
from collections import Counter;
import itertools;
import pprint;
import sys
from sklearn.decomposition import PCA


fileName='cho.txt' #Enter input file path here
test= np.loadtxt(fileName, delimiter="\t",dtype='str').shape[1]

k=5
iterations=1
maxIterations=100;
initial = np.loadtxt(fileName, delimiter="\t",dtype='str')
col=initial.shape[1]
rows=initial.shape[0]
firstcol= initial[:,0]
ground_truth= initial[:,1]
#print(ground_truth)
matrix = np.loadtxt(fileName, delimiter="\t",usecols=range(2,test)) # matrix without first 2 columns


def eucli_dist(matrix1,matrix2):
	distance=np.sqrt(np.sum((matrix1-matrix2)**2))
	distance = np.linalg.norm(matrix1 - matrix2)
	

	return distance

row=len(matrix)

#idx = np.random.randint(rows, size=k)
idx=[108,9,6,376,379]# ENTER INITAL CLUSTER CENTROID IDS HERE
print("Initial Centers: " ,idx)
idx[:] = [x - 1 for x in idx]

t=(row,row)
ground_matrix= np.zeros(t)

for i in range(0,row): 
	for j in range(0,row):
		if(initial[i][1]==initial[j][1]):
			ground_matrix[i][j]=1
		else:
			ground_matrix[i][j]=0

def metric(matrixa, matrixb):
	m11=0; m10=0; m01=0; m00=0;
	for i in range(rows):
		for j in range(rows):
			if matrixa[i][j]==matrixb[i][j]==1:
				m11=m11+1
			elif matrixa[i][j]==matrixb[i][j]==0:
				m00=m00+1
			elif matrixa[i][j]==1 and matrixb[i][j]==0:
				m10=m10+1
			elif matrixa[i][j]==0 and matrixb[i][j]==1:
				m01=m01+1
	rand = float(m11 +m00)/(m00+m11+m01+m10)
	jak=float (m11)/(m11+m10+m01)
	print("RAND :" ,rand)
	print( "JACCARD :" ,jak)



centers=matrix[idx,:]
centers=centers.tolist()



old_centers=[[] for i in range(k)]

while old_centers!=centers and iterations <=maxIterations:
	clusters={}
	cluster_ids={}
	c=0;
	for i in centers:
		
		clusters[c] = []
		cluster_ids[c]=[]
		
		c=c+1;
	
	result_a = list(range(len(matrix)))
	for i in range(len(matrix)):
		closest=sys.maxsize;pos=0;c=0;
		for j in centers:
			diff=eucli_dist(matrix[i],j)
			#print(diff)
			if diff<closest:
				pos=c
				closest=diff;

			c=c+1
		clusters[pos].append(matrix[i])
		cluster_ids[pos].append(initial[i][0])
		result_a[i] =pos
	for i in range(0,k):
		old_centers[i]=centers[i]
	
	
	for i in range(0,k):

		
		centers[i]=np.mean(clusters[i],axis=0).tolist()
	iterations=iterations+1

for i in range(0,k):
	print ("cluster" ,i, " has ",cluster_ids[i])
for i in range(0,k):
	print ("cluster" ,i, " has ",len(clusters[i]))
	
tp=(rows,col)
tp_gen=(rows,col-2)

clus_matrix= np.zeros(tp)
clus_col=np.array(result_a)
clus_col=np.transpose(clus_col)
clus_col=np.reshape(clus_col,(row,1))
clus_gen=[]
id=0
for i in initial:
	clus_gen.append(i[2:])
clus_gen = np.array(clus_gen)


clus_matrix=np.concatenate((clus_col,clus_gen),axis=1)


real_matrix=np.zeros((rows,rows))


for i in range(0,row):
	for j in range(0,row):
		if(clus_matrix[i][0]==clus_matrix[j][0]):
			real_matrix[i][j]=1
		else:
			real_matrix[i][j]=0

def pca2(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)

metric(real_matrix,ground_matrix)
pca_datset=pca2(matrix)
real_labels=real_matrix[:,0]
real_labels=np.reshape(real_labels,(real_labels.shape[0],1))
#print("cluster cols")
#print(clus_col)


result=np.zeros((rows,3))
result=np.concatenate((pca_datset,clus_col),axis=1)

#print(real_labels)
df_kmeans=pd.DataFrame(result)
df_kmeans.to_csv("cho_kmeans.csv")



