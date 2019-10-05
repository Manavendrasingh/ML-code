import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Mall_Customers.csv')
dataset.isnull().sum()
pd.plotting.scatter_matrix(dataset)
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,1] = lab.fit_transform(X[:,1])
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.ensemble import RandomForestClassifier
lin = RandomForestClassifier(max_depth = 5)
lin.fit(X_train,y_train)
lin.score(X_test,y_test)
#not complete efficiency is not good




