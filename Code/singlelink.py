import numpy as np
import sys
from sklearn.decomposition import PCA
import pandas as pd

sys.setrecursionlimit(1500)

#Loading the giving data set
file_name= np.loadtxt('new_dataset_2.txt', delimiter="\t",dtype=float)         
data=np.array(file_name)
rows,columns=data.shape
gene_id=[]
dis_mat=np.zeros((rows,rows))

number_of_clusters=3 #defining number of clusters

ground_truth=np.zeros((rows,rows))

for i in range(rows):
	for j in range(rows):
		if data[i][1]==data[j][1]:
			ground_truth[i][j]=1
		else:
			ground_truth[i][j]=0
#print(ground_truth)

for i in range(0,rows):
	gene_id.append([(int(data[i][0])-1)])
	
#print(gene_id)

def eucli_dist(matrix1,matrix2):

	distance = np.linalg.norm(matrix1-matrix2)
	return distance

def distance_matrix_calc(data):
	
	for i in range(0,rows):
		for j in range(0,rows):
			dis_mat[i][j]=eucli_dist(data[i][2:],data[j][2:])
	
	#print (dis_mat)	
	return dis_mat #the distance matrix is being returned


def min_value_calc(distance_matrix):

	r,c=distance_matrix.shape
	#min_value=distance_matrix[distance_matrix!=0.0].min()
	
	min_value=sys.maxsize
	x_min=-1
	y_min=-1
	for i in range(0,r):
	 	for j in range(0,r):
	 		if i!=j and min_value>distance_matrix[i][j]:
	 			min_value=distance_matrix[i][j]
	 			x_min=i
	 			y_min=j

	#print (x_min,y_min)
	#Appending the 2 new gene_id clusters together and later deleting the cluster which has been appended with the  cluster

	gene_id[x_min].extend(gene_id[y_min])
	del gene_id[y_min]
	
	form_cluster(distance_matrix,x_min,y_min)

def form_cluster(distance_matrix,x_min,y_min):
	#print("form_cluster")
	r,c=distance_matrix.shape
	
	for i in range(0,r):
		if x_min == i or y_min == i:
			continue
		#updating the distance matrix	
		distance_matrix[i][x_min]=min(distance_matrix[i][x_min],distance_matrix[i][y_min])
		distance_matrix[x_min][i]=min(distance_matrix[x_min][i],distance_matrix[y_min][i])
	
	distance_matrix=np.delete(distance_matrix,y_min, 0)
	distance_matrix=np.delete(distance_matrix,y_min, 1)	
		
	if distance_matrix.shape!=(number_of_clusters,number_of_clusters):
		min_value_calc(distance_matrix)
	else:
		print("Final Distance Matrix for given clusters")
		print (distance_matrix)

#this function gives the genes the cluster numer to which it belongs
def final_clusters(genes):
	output = [None for i in range(rows)]
	for i in range(len(genes)):
		for j in genes[i]:
			output[j] = i
	#print(output)		
	return output 

def generate_matrix(mat):
	A = [[0 for i in range(len(mat))] for i in range(len(mat))]
	for i in range(len(mat)):
		for j in range(len(mat)):
			if mat[i] == mat[j]:
				A[i][j] = 1
				A[j][i] = 1
			
	return A

#Calculating the Jaccard and Rand Index
def metric(matrix_a,matrix_b):
	M11=0;M10=0;M01=0;M00=0
	for i in range(rows):
		for j in range(rows):
			if matrix_a[i][j]==matrix_b[i][j]==1:
				M11=M11+1 
			elif matrix_a[i][j]==matrix_b[i][j]==0:
				M00=M00+1
			elif matrix_a[i][j]==1 and matrix_b[i][j]==0:
				M10=M10+1
			elif matrix_a[i][j]==0 and matrix_b[i][j]==1:
				M01=M01+1
	RAND=(M11+M00)/(M00+M11+M01+M10)
	JACCARD=M11/(M11+M10+M01)

	print("Rand Coefficient is:", RAND)
	print("Jaccard Coefficient is:", JACCARD)

#Calculating the PCA
def pca2(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)

			
###########--------MAIN SCRIPT---------------########

DM=distance_matrix_calc(data)
min_value_calc(DM)
print("Final Gene list")
print(gene_id)
#print(len(gene_id[0]))
temp = final_clusters(gene_id)
#mata = generate_matrix(data[:,1])
mat_gene_cluster = generate_matrix(temp)
metric(ground_truth,mat_gene_cluster)
pca_dataset=pca2(data[:,2:])

myarray = np.asarray(temp)
myarray=np.reshape(myarray,(myarray.shape[0],1))

result=np.zeros((rows,3))
result=np.hstack((pca_dataset,myarray))

df_hac=pd.DataFrame(result)
df_hac.to_csv("cho_hac_3.csv")



