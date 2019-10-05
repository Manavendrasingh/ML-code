import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
datasettest = pd.read_csv('poker-hand-testing.csv')#testing data 
X_test = datasettest.iloc[:, 0:10].values
y_test = datasettest.iloc[:,-1].values

dataset = pd.read_csv('poker-hand-training-true.csv')#training dataset 

X_train = dataset.iloc[:, 0:10].values
y_train = dataset.iloc[:,-1].values

pd.plotting.scatter_matrix(dataset)

dataset.isnull().sum() 

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 20)
dt.fit(X_train,y_train)
dt.score(X_test,y_test)
y_pred = dt.predict(X_test)
y_pred = pd.DataFrame(y_pred)#convert prediction into dataframe
y_pred[0].value_counts()#count different observation in y_pred[0]
y_test = pd.DataFrame(y_test)#converting y_test into dataframe
y_test[0].value_counts()#count different observation in y_pred[0]
