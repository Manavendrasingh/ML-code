import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#classification problem
dataset = pd.read_csv('sal.csv',names = ['age',
                                                  'workclass',
                                                  'fnlwgt',
                                                  'education',
                                                  'education-num',
                                                  'marital-status',
                                                  'occupation',
                                                  'relationship',
                                                  'race',
                                                  'gender',
                                                  'capital-gain',
                                                  'capital-loss',
                                                  'hours-per-week',
                                                  'native-country',
                                                  'salary'],na_values = ' ?')
X = dataset.iloc[:,0:14].values
y = dataset.iloc[:,-1].values
pd.plotting.scatter_matrix(dataset)

dataset.isnull().sum()
#handling missing values 

temp = pd.DataFrame(X[:, [1, 6, 13]])
temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()

temp[0] = temp[0].fillna(' Private')
temp[0] = temp[0].fillna(' Prof-specialty')
temp[0] = temp[0].fillna(' United-States')

X[:, [1, 6, 13]] = temp
del(temp)

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,1] = lab.fit_transform(X[:,1])
X[:,3] = lab.fit_transform(X[:,3])
X[:,5] = lab.fit_transform(X[:,5])
X[:,6] = lab.fit_transform(X[:,6].astype(str))
X[:,7] = lab.fit_transform(X[:,7])
X[:,8] = lab.fit_transform(X[:,8])
X[:,9] = lab.fit_transform(X[:,9])
X[:,13] = lab.fit_transform(X[:,13].astype(str))

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 10])
X =one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X = sts.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)






