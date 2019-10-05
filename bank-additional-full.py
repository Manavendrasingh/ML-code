import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#classificatio problem
dataset = pd.read_csv('bank-additional-full.csv',sep = ";",na_values = 'unknown')
pd.plotting.scatter_matrix(dataset)

X = dataset.iloc[:,[0,1,2,3,5,6,11,12,13,15,16,17,18,19]].values



dataset.isnull().sum()
temp = pd.DataFrame(X[:,[1 ,2 ,3 ,4 ,5 ]])
temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()
temp[3].value_counts()
temp[4].value_counts()




temp[0] = temp[0].fillna('admin.')
temp[1] = temp[1].fillna('married')
temp[2] = temp[2].fillna('university.degree')
temp[3] = temp[3].fillna('yes')
temp[4] = temp[4].fillna('no')


X[:,[1 ,2 ,3 ,4 ,5]] = temp

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,1] = lab.fit_transform(X[:,1])
X[:,2] = lab.fit_transform(X[:,2])
X[:,3] = lab.fit_transform(X[:,3])
X[:,4] = lab.fit_transform(X[:,4])
X[:,5] = lab.fit_transform(X[:,5].astype(str))


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1 ,2 ,3 ,4 ,5])
X = one.fit_transform(X)





