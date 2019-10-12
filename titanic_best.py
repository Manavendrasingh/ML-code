import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


submission = pd.read_csv('gender_submission.csv')
test = pd.read_csv('test.csv')
train= pd.read_csv('train.csv')

corr  = train.corr()
corr['PassengerId'].sort_values()
corr['Survived'].sort_values()
del(corr)

#data preprocessing start here
train.drop('Name',axis = 'columns', inplace = True)
train.drop('Ticket',axis = 'columns', inplace = True)
train.drop('Cabin',axis = 'columns', inplace = True)
train.drop('Embarked',axis = 'columns', inplace = True)

test.drop('Name',axis = 'columns', inplace = True)
test.drop('Ticket',axis = 'columns', inplace = True)
test.drop('Cabin',axis = 'columns', inplace = True)
test.drop('Embarked',axis = 'columns', inplace = True)
train.isnull().sum()
test.isnull().sum()



X_train  =  train.iloc[:,:].values
X_train = np.delete(X_train,1,axis  = 1)
y_train = train.iloc[:,1].values

X_test = test.iloc[:,:].values


from sklearn.preprocessing import Imputer
imp = Imputer()
X_train[:,[3]] = imp.fit_transform(X_train[:,[3]])
X_test[:,[3]] = imp.fit_transform(X_test[:,[3]])
X_test[:,[6]] = imp.fit_transform(X_test[:,[6]])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X_train[:,2] = lab.fit_transform(X_train[:,2])
lab.classes_
X_test[:,2] = lab.fit_transform(X_test[:,2])
lab.classes_


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 8)
dt.fit(X_train,y_train)
dt.score(X_train,y_train)
y_pred = dt.predict(X_test)

y_pred = y_pred.reshape(-1,1)
my_submission = test.iloc[:,0].values
my_submission = my_submission.reshape(-1,1)
my_submission = np.concatenate((my_submission,y_pred),axis = 1)
#this is complete

















