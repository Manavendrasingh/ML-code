import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv('housing.csv')

pd.plotting.scatter_matrix(dataset)#ye kyu plot kiya tha ?????

plt.scatter(dataset['households'],dataset['total_rooms'])#ye kyu plot kiya tha ?????
plt.show()

X = dataset.iloc[:,[0,1,2,3,4,5,6,7,9]].values
y = dataset.iloc[:,8].values


dataset.isnull().sum()# isnull donot work on object

from sklearn.preprocessing import Imputer#handing missing values
imp = Imputer(strategy = 'median')
X[:,[4]] = imp.fit_transform(X[:,[4]])


from sklearn.preprocessing import LabelEncoder#replace categorical vaules by a number
lab = LabelEncoder()
X[:,8] = lab.fit_transform(X[:,8])

from sklearn.preprocessing import OneHotEncoder#solving dammy variable trap
one = OneHotEncoder(categorical_features = [8])
X = one.fit_transform(X)
X = X.toarray()


from sklearn.preprocessing import StandardScaler#feature scalling 
lab = StandardScaler()
X = lab.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)


from sklearn.linear_model import LinearRegression#implement linear regression
sc =LinearRegression()
sc.fit(X_train,y_train)

sc.score(X_test,y_test)
sc.score(X_train,y_train)
sc.score(X,y)

y_pred = sc.predict(X)

sc.coef_
sc.intercept_
























