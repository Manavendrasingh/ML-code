import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.datasets import make_blobs
x,y = make_blobs(n_samples = 300,centers = 6,cluster_std = 0.6 )#




plt.scatter(x[:,0],x[:,1])
plt.show()


import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(x,method = 'ward'))


from sklearn.cluster import AgglomerativeClustering
hca = AgglomerativeClustering(n_clusters =  2)
y_pred = hca.fit_predict(x)

plt.scatter(x[:,0],x[:,1])
plt.show()

import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(x,method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
hca = AgglomerativeClustering(n_clusters = 2)
y_pred = hca.fit_predict(x)

plt.scatter(x[y_pred == 0,0],x[y_pred ==0,0])
plt.scatter(x[y_pred == 1,0],x[y_pred ==1,1])
plt.show()


