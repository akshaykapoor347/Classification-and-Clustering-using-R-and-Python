# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:38:44 2018

@author: Akshay
"""
#import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import tree
from sklearn.decomposition import PCA



iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

#Plotting sepal.length and sepal.width
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 

plt.figure(2, figsize=(8,6))
plt.clf()

#Plot training points

plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
plt.xlabel('Sepal Length')
plt.xlabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


#Plotting 3d to understand the interactions

fig = plt.figure(2, figsize=(8,6))
ax = Axes3D(fig, elev = -150, azim = 110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c = y, 
cmap = plt.cm.Set1, edgecolor = 'k', s = 40)

#Splitting data into test and training data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.7, random_state = 1)

#decision tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)
#Accuracy of 93.33 percent



#Using randomForest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)
#Accuracy of 95.23 percent better than decision tree!

#Clustering

#using K means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10,
                random_state = 0)

y_kmeans = kmeans.fit_predict(X)

#plotting
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red' ,
            label = 'setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue' ,
            label = 'versicolor')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green' ,
            label = 'virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s = 100, c = 'black', label = 'centroid' )

plt.legend()

#Using Dbscan

from sklearn.cluster import DBSCAN
dbscan = DBSCAN()


DBSCAN(metric = 'euclidean', min_samples=5)

dbscan.fit(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',
    marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',
    marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',
    marker='*')
        
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])

plt.title('DBSCAN finds 2 clusters and noise')
plt.show()
