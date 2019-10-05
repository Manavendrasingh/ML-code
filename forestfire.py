import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('forestfires.csv')
pd.plotting.scatter_matrix(dataset)

X = dataset.iloc[:,0:12].values
y = dataset.iloc[:,-1].values

dataset.isnull().sum()
dataset.info()
temp = pd.DataFrame(X[:,[2,3]])
temp_month = pd.get_dummies(temp[0])
temp_day = pd.get_dummies(temp[1])
del(temp)

X = np.append(X,temp_month,axis = 1)
X = np.append(X,temp_day,axis = 1)
X = np.delete(X,2,axis =1)
X = np.delete(X,2,axis =1)
del(temp_month,temp_day)

temp = pd.DataFrame(X[:,:])

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X = st.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)





from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

from sklearn.ensemble import RandomForestRegressor
ran = RandomForestRegressor(n_estimators = 5)
ran.fit(X_train,y_train)
ran.score(X_train,y_train)
#this is complete




