ReadMe: K-Means Clustering Algorithm
---------------------------------------------------------------------------------------------------------------------
File to be run : kmeans1.py

1. Assign the file path of input file (cho.txt or iyer.txt) to fileName
	a. fileName='/home/hp/Desktop/project2/cho.txt'
2. Set the number of clusters to K
	a. k=5
3. Set the number of maximum iterations to maxIterations
	a. maxIterations=100
4. Enter initial centroids in idx
	a. idx=[108,9,6,376,379]
5. Set the name of output file to be used as input for plots
6. Now open the kmeans.ipynb file and give the name of the output file of kmeans1.py in the following statement:
	data = read.csv(name_of_csv_file)

README : Hierarchical Agglomerative Clustering

---------------------------------------------------------------------------------------------------------------------
File to be run : singlelink.py

1. Write the file name which has to be loaded for the implementation in the variable ‘file_name’ like given below:
	a. file_name= np.loadtxt('new_dataset_2.txt', delimiter="\t",dtype=float)
2. Set the value of the variable ‘number_of_clusters’ to the number of clusters needed in HAC	
	a. number_of_clusters=3
3. At the end of the code, give the name to the csv file which is to be used for plotting graphs in R in the jupyter notebook 
	a. eg . df_hac.to_csv(newdset_hac_3.csv)
4. Now open the hac.ipynb file and give the name(or the path) of the csv file  in the statement :
	a. data=read.csv(name_of_csv_file.csv).
5. Now run the R notebook (hac.ipynb) to get the PCA plots for the given data set giving the generated csv output file from the code.

README : DBSCAN ALGORITHM
---------------------------------------------------------------------------------------------------------------------
File to be run : dbscan.py

1. Assign the path of the file for which to implement dbscan algorithm in the ‘path’ variable.
	a. path=’iyer.txt’ or ‘cho.txt’ or filename
2. Set the value of epsilon and minpts.
3. Change the name of the output_file accordingly and run the code. A csv file of the reduced dimensions with new cluster values is formed.
4. Open the dbscan.ipynb file and change the name of the input csv file as per the given output_file name. 
	a. data = read.csv(name_of_csv_file)

ReadMe: K-Means Clustering Algorithm by MapReduce
---------------------------------------------------------------------------------------------------------------------

File to be run - runner.py
Driver file - runner.py
Mapper file- mapper.py
Reducer file - reducer.py

1. Perform the following steps in runner.py:
2. Assign input filename to inputFile variable  in runner.py
	a. inputFile=’cho.txt’
	b. Also change the filename in the following command which is called by the subprocess module to run mapper and reducer
	c. hadoop jar hadoop-streaming-2.7.4.jar -file mapper.py -mapper mapper.py -file reducer.py -reducer reducer.py -input cho.txt -output data-output
3. Enter the name of the file that will contains centroids to centroidDatafile
	a. centroidDatafile=’data.txt’
4. Set number of clusters in k
	a. k=5
5. Enter maximum number of iterations in maxIterations
	a. maxIterations=25
6. Enter initial centroids in idx
	a. idx=[108,9,6,376,379]
7. In the mapper.py file, enter file path containing initial centroids at centroidDataFile
	a. centroidDataFile='/home/hp/Desktop/hduser/data.txt'
8. After making the above changes, run the runner.py file

