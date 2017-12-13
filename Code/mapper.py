#! /usr/bin/env python
import sys
import numpy as np

centroidDataFile='/home/hp/Desktop/hduser/data.txt' # enter file containing centroid datas
centroidData = np.loadtxt(centroidDataFile, delimiter="\t")
centroidData=centroidData[:,1:]


def eucli_dist(matrix1,matrix2):
	#print("matrix diff " , matrix1-matrix2)
	#print("matrix2" , matrix2)
	#distance=np.sqrt(np.sum(matrix1-matrix2)**2)
	#print("distance", distance)
	distance = np.linalg.norm(matrix1 - matrix2)
	return distance
for line in sys.stdin:
	#l=[]
	l=line.strip()
	if not l:
		continue
	l=l.split("\t")
	#print(repr("CHECKING"))
	#print(repr(l))
	l=np.array(l)
	#print(l[0])
	l = l.astype(np.float)
	l=l[2:]
	closest=sys.maxint;pos=0;c=0;
	for j in centroidData:
		j=j.astype(np.float)
		#k=j[2:]
		diff=eucli_dist(l,j)
		#print(diff)
		if diff<closest:

			#print("Inside if")
			pos=c
			#print("diff", diff)
			closest=diff;

		c=c+1
	#print( pos, l);
	pos=pos+1
	st='\t'.join((map(str,l)))
	print '%s\t%s' %(pos,st)