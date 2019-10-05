import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
dataset = pd.read_csv('weatherHistory.csv')
#pd.plotting.scatter_matrix(dataset)

X = dataset.iloc[:,0:11].values
y = dataset.iloc[:,-1].values
y = y.reshape(-1,1)

temp  = pd.DataFrame(X[:,0])
Year = []
Month = []
Date = []
Time = []
Formate = []
#to preprocess the column no 0 in X
for i in range(96453):
    date = re.sub('[^0-9:.+]',' ',temp[0][i])
    year,month,da,time,formate = date.split()
    #print(year,month,da,time,formate)
    Year.append(int(year))
    Month.append(month)
    Date.append(da)
    Time.append(time)
    Formate.append(formate)
array1 = np.array(Year) 
array2 = np.array(Month)
array3 = np.array(Date)
array4 = np.array(Time)
array5 = np.array(Formate)
del(temp)

array1 = array1.reshape(-1,1)
array2 = array2.reshape(-1,1)
array3 = array3.reshape(-1,1)
array4 = array4.reshape(-1,1)
array5 = array5.reshape(-1,1)
#concat these array with X
X = np.concatenate((X,array1),axis = 1)
X = np.concatenate((X,array2),axis = 1)
X = np.concatenate((X,array3),axis = 1)
X = np.concatenate((X,array4),axis = 1)
X = np.concatenate((X,array5),axis = 1)
#deleting the fist column because we split it into fifferent columns 
X = np.delete(X,0,axis = 1)

#make seprate column of data and month ,year,time ,timezone column 
temp = pd.DataFrame(X[:,10:15])
temp_0 = pd.get_dummies(temp[0])
temp_1 = pd.get_dummies(temp[1])
temp_2 = pd.get_dummies(temp[2])
temp_3 = pd.get_dummies(temp[3])
temp_4 = pd.get_dummies(temp[4])
del(temp)

#after the get dummiesing delete the data and month ,year,time ,timezone
X = np.delete(X,10,axis = 1)
X = np.delete(X,10,axis = 1)
X = np.delete(X,10,axis = 1)
X = np.delete(X,10,axis = 1)
X = np.delete(X,10,axis = 1)
#apeend the the column that fomed by getdumminesing into X
X = np.append(X,temp_0,axis = 1)
X = np.append(X,temp_1,axis = 1)
X = np.append(X,temp_2,axis = 1)
X = np.append(X,temp_3,axis = 1)
X = np.append(X,temp_4,axis = 1)
temp = pd.DataFrame(X[:,:])


from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,0] = lab.fit_transform(X[:,0])
X[:,1] = lab.fit_transform(X[:,1].astype(str))
y = lab.fit_transform(y)
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one  = OneHotEncoder(categorical_features = [0,1])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.ensemble import RandomForestClassifier
ran = RandomForestClassifier(n_estimators=5)
ran.fit(X_train,y_train)
ran.score(X_test,y_test)







