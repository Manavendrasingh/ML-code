import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

dataset = pd.read_csv('MissingMigrants-Global-2019-03-29T18-36-07.csv',na_values = ['Uncategorized','Unknown (skeletal remains)']
)
pd.plotting.scatter_matrix(dataset)

#to check orrelation
corr = dataset.corr()
corr["Source Quality"].sort_values()
dataset.info()
del(corr)
#delete web id corr because it has negative corr
dataset.drop(["Web ID"],axis = "columns", inplace = True)

X = dataset.iloc[:,0:18].values
y = dataset.iloc[:,-1].values
temp = pd.DataFrame(X[:,[1,14]])
Date = []
x_corr = []
y_corr = []
#making different column of dataclumn and coradinate cloumn
for i in range(5333):
    Reported_Date = re.sub('[^a-zA-z0-9]',' ',temp[0][i])
    month,date,year = Reported_Date.split()  
    Date.append(int(date))
    Location_co_ordinate = re.sub('[^a-zA-Z0-9.]',' ',temp[1][0])
    
    x_co,y_co = Location_co_ordinate.split()
    x_corr.append(float(x_co))
    y_corr.append(float(y_co))
    
 #convert list into array   
array1 = np.array(Date)
array2 = np.array(x_corr)
array3 = np.array(y_corr)
#reshape the array
array1 = array1.reshape(-1,1)
array2 = array2.reshape(-1,1)
array3 = array2.reshape(-1,1)
#add in X
X = np.concatenate((X,array1),axis = 1)
X = np.concatenate((X,array2),axis = 1)
X = np.concatenate((X,array3),axis = 1)
del(temp,x_co,y_co,x_corr,y_corr,Date,month,date,year,array1,array2,array3,Reported_Date, Location_co_ordinate)

#delete some cloumn which is not in used 
X = np.delete(X,1,axis = 1)
X = np.delete(X,11,axis = 1)
X = np.delete(X,11,axis = 1)
X = np.delete(X,11,axis = 1)
X = np.delete(X,12,axis = 1)

temp2 = pd.DataFrame(X[:,:])
temp2.isnull().sum()
dataset.info()
dataset.isnull().sum()
temp2.info()

#get dummies the data related columns 
temp = pd.DataFrame(X[:,[1,2,13]])
temp_1 = pd.get_dummies(temp[0])
temp_2 = pd.get_dummies(temp[1])
temp_13 = pd.get_dummies(temp[2])

X = np.append(X,temp_1,axis = 1)
X = np.append(X,temp_2,axis = 1)
X = np.append(X,temp_13,axis = 1)

X = np.delete(X,1,axis = 1)
X = np.delete(X,1,axis = 1)
X = np.delete(X,11,axis = 1)
del(temp,temp_1,temp_2,temp_13)

#handing missing values in numeric data
from sklearn.preprocessing import Imputer
imp = Imputer()
X[:,1:8] = imp.fit_transform(X[:,1:8])

temp = pd.DataFrame(X[:,[8,9,10]])
temp.isnull().sum()

temp[0].value_counts()
temp[0] = temp[0].fillna('Drowning')


temp[1].value_counts()
temp[1] = temp[1].fillna('Central America to US')

temp[2].value_counts()
temp[2] = temp[2].fillna('Northern Africa')
X[:,[8,9,10]] = temp

#handling categorical values 
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,0] = lab.fit_transform(X[:,0])
X[:,8] = lab.fit_transform(X[:,8])
X[:,9] = lab.fit_transform(X[:,9])
X[:,10] = lab.fit_transform(X[:,10])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [0,8,9,10])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X = sts.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.ensemble import RandomForestClassifier
ran = RandomForestClassifier(n_estimators=5)
ran.fit(X_train,y_train)
ran.score(X_test,y_test)
#this is complete















