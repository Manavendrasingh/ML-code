import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.datasets import fetch_mldata
#dataset = fetch_mldata('MNIST original')

from sklearn.datasets import load_digits
dataset = load_digits()

X = dataset.data
y = dataset.target

some_digit = X[58]
some_digit_image = some_digit.reshape(8, 8)#reshape(8,8) fomred a metric of 8*8 by 64*1 metr

plt.imshow(some_digit_image)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
log_reg.fit(X_train,y_train)
log_reg.score(X,y)
log_reg.score(X_train,y_train)
log_reg.score(X_test,y_test)

knn.fit(X_train,y_train)
knn.score(X,y)
knn.score(X_train,y_train)
knn.score(X_test,y_test)





from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 13)
dtf.fit(X, y)

dtf.score(X,y)

from sklearn.tree import export_graphviz
export_graphviz(dtf,out_file = "tree.dot")


import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
