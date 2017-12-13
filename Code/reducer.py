#! /usr/bin/env python
from operator import itemgetter
import sys
import numpy as np


current=None

sum_g = []

for line in sys.stdin:

	#print(line)
	l=line.strip()


	temp=l.split("\t")
	if not temp:
		continue

	cluster=int(temp[0])
	#print(cluster)
	gene=np.array(temp[1:], dtype = np.float)
	#gene=gene.astype(np.float)


	if len(sum_g)==0:
		current=cluster

	if current==cluster:
		sum_g.append(gene)
		#print("cluster when current is present",cluster)
	else:

		avg=np.mean(sum_g, axis=0)
		#print("average",avg)
		st='\t'.join((map(str,avg)))
		print ('%s\t%s' %(current,st))
		current=cluster
		gene_count=1
		sum_g=[gene]

	#print(gene)
	


if current==cluster:
	avg=np.mean(sum_g, axis=0)
	st='\t'.join((map(str,avg)))
	print ('%s\t%s' %(current,st))
	#print '%s\t%s' % (cluster,current_count)