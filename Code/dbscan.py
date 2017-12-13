import numpy as np
import sys
from sklearn.decomposition import PCA
import pandas as pd
import csv

path='iyer.txt'
test=np.loadtxt(path, delimiter="\t",dtype='float')
cols=test.shape[1]
rows=test.shape[0]
print(rows,cols)
z=np.zeros((rows,2),dtype=test.dtype)
test=np.concatenate((test,z),axis=1)
#c_index will have the updated clusters for the gene_ids, it is the extra column added to the original data matrix
c_index=cols 
t=np.asmatrix(test[:,c_index])
t=np.transpose(t)
#v is an extra column added to the original data matrix which keeps a track if the gene_id has been visited or not
v=cols+1
epsilon=1.21
minpts=3
cluster_f=0
temp=np.zeros((rows,(cols-2)))
temp=test[:,2:(cols-1)]
ground_truth=np.zeros((rows,rows))
cluster_matrix=np.zeros((rows,rows))
distanceMat=np.zeros((rows,rows))

def eucli_dist(m1,m2):
    distance=np.sqrt(np.sum((m1-m2)**2))
    return distance
def calDistanceMatrix(test):
	for i in range(0,rows):
		for j in range(0,rows):
			distanceMat[i][j]=eucli_dist(test[i][2:test.shape[1]-2],test[j][2:test.shape[1]-2])
	return distanceMat


def dbscan(test):
    cluster=0
    for i in range(0,rows):
        
        if test[i][v]==0:
            test[i][v]=1
            neighbours=regionQuery(i)
            if len(neighbours)<minpts:
                test[i][v]=1
            else:
                cluster=cluster+1
                expandCluster(i,neighbours,cluster)
				
    cluster_f=cluster
    print("Number of clusters formed ",cluster_f)
    return cluster_f
	
def expandCluster(p,neighbours,cluster):
    test[p][c_index]=cluster

    while len(neighbours)>0:
        i=neighbours.pop()
        if test[i][v]==0:
            test[i][v]=1
            neighbours2=regionQuery(i)
            if len(neighbours2)>=minpts:
                neighbours = neighbours.union(neighbours2)
        if test[i][c_index]==0:
            test[i][c_index]=cluster

def regionQuery(p):
	l=set()
	for i in range(0,rows):
		if distanceMat[p][i]<=epsilon:
			l.add(i)
	return l

def jaccard(matrix1,matrix2):
    m11=0; m10=0; m01=0; m00=0;
	
    for i in range(rows):
        for j in range(rows):
            
            if matrix1[i][j]==matrix2[i][j] and matrix2[i][j]==1:
                m11=m11+1
            elif matrix1[i][j]==matrix2[i][j] and matrix2[i][j]==0:
                m00=m00+1
            elif matrix1[i][j]==1 and matrix2[i][j]==0:
                m10=m10+1
            elif matrix1[i][j]==0 and matrix2[i][j]==1:
                m01=m01+1
	

    rand=(m11+m00)*1.0/(m00+m11+m01+m10)
    print("Rand index",rand)
    jac=m11*1.0/(m11+m10+m01)
    
    print("Jaccard coefficient",jac)




mat=calDistanceMatrix(test)

cluster_f=dbscan(test)

for i in range(0,rows):
    for j in range(0,rows):
        if(test[i][1]==test[j][1]):
            ground_truth[i][j]=1
        else:
            ground_truth[i][j]=0
for i in range(0,rows):
    for j in range(0,rows):
        if test[i][c_index]==test[j][c_index]:
            cluster_matrix[i][j]=1
        else:
            cluster_matrix[i][j]=0
jaccard(ground_truth,cluster_matrix)


def pca2(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)

ans=pca2(temp)
result=np.zeros((rows,3))
result=np.concatenate((ans,t),axis=1)

print(result)
gene_clusters={}
for i in range(0,cluster_f+1):
	gene_clusters[i]=[]

for i in range(rows):
	if test[i][c_index] in gene_clusters:
		gene_clusters[test[i][c_index]].append(test[i][0])
	else:
		gene_clusters[test[i][c_index]]=test[i][0]

#gene_clusters has the final clusters that are formed after the dbscan is implemented
print (gene_clusters)
	
output_file="iyer.csv"
df=pd.DataFrame(result)
df.to_csv(output_file)
