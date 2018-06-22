# Analysis

## Overview

This section identifies distinct weather patterns using k-means clustering in the data collected from a weather station in San Diego, CA . First, we determine the optimal number of clusters using an elbow plot.  Then we find cluster centers for the optimal number of clusters to identify weather patterns corresponding to each cluster.

The [previous section](https://eagronin.github.io/weather-clustering-spark-prepare/) explores, cleans and scales the data to prepare them for the clustering analysis.

The [next section](https://eagronin.github.io/weather-clustering-spark-report/) reports and interprets the results.

## K-Means Clustering

The k-means algorithm requires that the number of clusters (k) has to be specified. To determine a good value for k, we will use the “elbow” method. This method applies k-means using different values for k and calculating the within-cluster sum-of-squared error (WSSE).  This process can be compute-intensive, because k-means is applied multiple times. Therefore, for creating the elbow plot we use only a subset of the dataset. Specifically, we keep every third row from the dataset and drop all the other rows:

```python
from itertools import cycle, islice
from math import sqrt
from numpy import array
from pandas.tools.plotting import parallel_coordinates
from pyspark.ml.clustering import KMeans as KM
from pyspark.mllib.linalg import DenseVector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

scaledData = scaledData.select('features', 'rowID')
elbowset = scaledData.filter((scaledData.rowID % 3) == 0).select('features')
elbowset.persist()
```

The persist() method in the last row is used to keep the resulting dataset in memory for faster processing.  

Next, we define several functions that we will need for creating the elbow plot and parallel coordinate plots of cluster centroids.

WSSE discussed above is calculated using the following function: 

```python
def computeCost(featuresAndPrediction, model):
    allClusterCenters = [DenseVector(c) for c in model.clusterCenters()]
    arrayCollection   = featuresAndPrediction.rdd.map(array)

    def error(point, predictedCluster):
        center = allClusterCenters[predictedCluster]
        z      = point - center
        return sqrt((z*z).sum())
    
    return arrayCollection.map(lambda row: error(row[0], row[1])).reduce(lambda x, y: x + y)
```

The function below iterates over the number of clusters to determine WSSE for each number of clusters to create the elbow plot:

```python
def elbow(elbowset, clusters):
	wsseList = []	
	for k in clusters:
		print("Training for cluster size {} ".format(k))
		kmeans = KM(k = k, seed = 1)
		model = kmeans.fit(elbowset)
		transformed = model.transform(elbowset)
		featuresAndPrediction = transformed.select("features", "prediction")

		W = computeCost(featuresAndPrediction, model)
		print("......................WSSE = {} ".format(W))

		wsseList.append(W)
	return wsseList
```

The function below plots WSSE against the number of clusters:

```python
def elbow_plot(wsseList, clusters):
	wsseDF = pd.DataFrame({'WSSE' : wsseList, 'k' : clusters })
	wsseDF.plot(y='WSSE', x='k', figsize=(15,10), grid=True, marker='o')
```

The next function converts cluster centers determined by the the k-means algorithm to a pandas dataframe in order to plot cluster centers in matplotlib (Spark dataframes cannot be plotted using matplotlib): 

```python
def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P
```

Finally, the function below plots cluster centers:

```python
def parallel_plot(data, P):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(P)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
```    
  
The code in the remainder of this section calls the functions above to create an elbow plot and fit the k-means algorithm for the chosen number of clusters.  

The following code calculates WSSE for each number of clusters ranging from 2 to 30:

```
clusters = range(2, 31)
wsseList = elbow(elbowset, clusters)
```

This code generates the following output, which is going to be used as input into the function that creates an elbow plot:

```
Training for cluster size 2 
......................WSSE = 114993.13 
Training for cluster size 3 
......................WSSE = 104181.09
Training for cluster size 4 
......................WSSE = 94577.27
Training for cluster size 5 
......................WSSE = 87993.46 
Training for cluster size 6 
......................WSSE = 85084.23 
Training for cluster size 7 
......................WSSE = 81664.96 
Training for cluster size 8 
......................WSSE = 78397.76 
Training for cluster size 9 
......................WSSE = 76599.60 
Training for cluster size 10 
......................WSSE = 74023.93 
Training for cluster size 11 
......................WSSE = 72772.61 
Training for cluster size 12 
......................WSSE = 70281.81 
Training for cluster size 13 
......................WSSE = 69473.53 
Training for cluster size 14 
......................WSSE = 68756.12 
Training for cluster size 15 
......................WSSE = 67394.28 
Training for cluster size 16 
......................WSSE = 66698.44 
Training for cluster size 17 
......................WSSE = 64559.11 
Training for cluster size 18 
......................WSSE = 63205.19 
Training for cluster size 19 
......................WSSE = 62368.485 
Training for cluster size 20 
......................WSSE = 62444.64 
Training for cluster size 21 
......................WSSE = 61265.25 
Training for cluster size 22 
......................WSSE = 60936.73 
Training for cluster size 23 
......................WSSE = 60109.64 
Training for cluster size 24 
......................WSSE = 60087.31 
Training for cluster size 25 
......................WSSE = 59084.88 
Training for cluster size 26 
......................WSSE = 58498.07 
Training for cluster size 27 
......................WSSE = 58215.79 
Training for cluster size 28 
......................WSSE = 58132.84 
Training for cluster size 29 
......................WSSE = 57426.52 
Training for cluster size 30 
......................WSSE = 57099.24 
```

By calling `elbow_plot(wsseList, clusters)` we create an elbowplot, which is presented in the [next section](https://eagronin.github.io/weather-clustering-spark-report/).

As we will discuss in the [next section](https://eagronin.github.io/weather-clustering-spark-report/), we choose the number of clusters to be 12 based on the elbow plot.

Once we have chosen the number of clusters using a scaled down version of the dataset, we will go back to the `scaledData` dataset to fit the k-means algorithm for 12 clusters using that dataset.  We again choose the persist() method to keep the datasest in memory for faster processing:

```python
scaledDataFeat = scaledData.select('features')
scaledDataFeat.persist()

kmeans = KMeans(k = 12, seed = 1)
model = kmeans.fit(scaledDataFeat)
transformed = model.transform(scaledDataFeat)

transformed.head(10)
```

The code above results in a dataset, in which each row is comprised of a point in the feature sapce and the cluster ID to which this point belongs:

```
[Row(features=DenseVector([-1.4846, 0.2454, -0.6839, -0.7656, -0.6215, -0.7444, 0.4923]), prediction=2),
 Row(features=DenseVector([-1.4846, 0.0325, -0.1906, -0.7656, 0.0383, -0.6617, -0.3471]), prediction=0),
 Row(features=DenseVector([-1.5173, 0.1237, -0.6524, -0.3768, -0.4485, -0.3723, 0.4084]), prediction=2),
 Row(features=DenseVector([-1.5173, 0.0629, -0.7468, -0.3768, -0.654, -0.4137, 0.3931]), prediction=2),
 Row(features=DenseVector([-1.5173, 0.1846, -0.8518, -0.0852, -0.8162, -0.2069, 0.3741]), prediction=2),
 Row(features=DenseVector([-1.5501, 0.1542, -0.6314, -0.7656, -0.4809, -0.7857, 0.1451]), prediction=2),
 Row(features=DenseVector([-1.5829, 0.1846, -0.8308, -1.0085, -0.6756, -1.0338, 0.1451]), prediction=2),
 Row(features=DenseVector([-1.6156, 0.1998, -0.8413, -0.3768, -0.7189, -0.4137, 0.5572]), prediction=2),
 Row(features=DenseVector([-1.6156, -0.0132, -0.9987, 0.255, -1.0109, 0.0411, 0.9121]), prediction=2),
 Row(features=DenseVector([-1.6156, -0.0436, -0.9987, 0.4008, -0.9568, 0.3305, 0.9502]), prediction=2)]
 ...
```

The fitted model is then used to determine the center measurement of each cluster:

```python
centers = model.clusterCenters()
centers
```

These cluster centers are shown below:

```
[array([-0.13720796,  0.6061152 ,  0.22970948, -0.62174454,  0.40604553, -0.63465994, -0.42215364]),
 array([ 1.42238994, -0.10953198, -1.10891543, -0.07335197, -0.96904335, -0.05226062, -0.99615617]),
 array([-0.63637648,  0.01435705, -1.1038928 , -0.58676582, -0.96998758, -0.61362174,  0.33603011]),
 array([-0.22385278, -1.06643622,  0.5104215 , -0.24620591,  0.68999967, -0.24399706,  1.26206479]),
 array([ 1.17896517, -0.25134204, -1.15089838,  2.11902126, -1.04950228,  2.23439263, -1.12861666]),
 array([-1.14087425, -0.979473  ,  0.42483303,  1.68904662,  0.52550171,  1.65795704,  1.03863542]),
 array([ 0.50746307, -1.08840683, -1.20882766, -0.57604727, -1.0367013 , -0.58206904,  0.97099067]),
 array([ 0.14064028,  0.83834618,  1.89291279, -0.62970435, -1.54598923, -0.55625032, -0.75082891]),
 array([-0.0339489 ,  0.98719067, -1.33032244, -0.57824562, -1.18095582, -0.58893358, -0.81187427]),
 array([-0.22747944,  0.59239924,  0.40531475,  0.6721331 ,  0.51459992,  0.61355559, -0.15474261]),
 array([ 0.3334222 , -0.99822761,  1.8584392 , -0.68367089, -1.53246714, -0.59099434,  0.91004892]),
 array([ 0.3051367 ,  0.67973831,  1.36434828, -0.63793718,  1.631528  , -0.58807924, -0.67531539])]
```

It is difficult to compare the cluster centers by just looking at these numbers. Therefore, we will visualize these numbers using parallel coordinates plots, which are used to visualize multi-dimensional data.  Each line plots the centroid of a cluster, and all of the features are plotted together on the same chart. Because the feature values were scaled to have mean = 0 and standard deviation = 1, the values on the y-axis of these parallel coordinates plots show the number of standard deviations from the mean.

The plots are created with matplotlib using a Pandas DataFrame.  Each row in the dataframe contains the cluster center coordinates and cluster label. We use the `pd_centers()` function described above to create the Pandas DataFrame:
 
```python
P = pd_centers(featuresUsed, centers)
```

Next step: [Results](https://eagronin.github.io/weather-clustering-spark-report/)
