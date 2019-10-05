import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:].values

pd.plotting.scatter_matrix(dataset)
X = np.delete(X,8,axis = 1)
y = dataset.iloc[:,8].values

dataset.isnull().sum()
temp = pd.DataFrame(X[:,:])
temp.isnull().sum()


from sklearn.preprocessing import Imputer
imp = Imputer()
X[:,[4]] = imp.fit_transform(X[:,[4]])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,-1] = lab.fit_transform(X[:,-1])


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [-1])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sdt = StandardScaler()
X = sdt.fit_transform(X)
'
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)
lin.score(X_test,y_test)
y_pred = lin.predict([[1.12214,-0.681889,-0.0155662,-0.353264,-0.384466,-1.5774,1.23046,0.505394,0.598304,0.255522,0.214161,0.315077,0.270731
]])
print(y_pred)

from sklearn.ensemble import RandomForestRegressor
rand = RandomForestRegressor(n_estimators = 5)
rand.fit(X_train,y_train)
rand.score(X_test,y_test)
#this is complete



#-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,	452600.0,1




