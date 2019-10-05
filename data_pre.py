import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('Data_Pre.csv')

dataset.info()

corr = dataset.corr()
corr["Diabetic"].sort_values()

pd.plotting.scatter_matrix(dataset)



X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,-1].values

plt.scatter(X[:,1],X[:,2])
plt.scatter(X[:,0],X[:,1])
plt.scatter(X[:,0],X[:,2])


dataset.isnull().sum()


from sklearn.preprocessing import Imputer
imp = Imputer()
X[:,[0,1]] = imp.fit_transform(X[:,[0,1]])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,2] = lab.fit_transform(X[:,2])



from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [0,1])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X = st.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nb.score(X_train,y_train)
nb.score(X_test,y_test)


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X_train,y_train)
log.score(X_test,y_test)
#this is complete






