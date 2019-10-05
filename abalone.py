import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#linear regression problem and classification also
dataset =pd.read_csv('abalone.csv',names = ['sex',
                                             'lenght',
                                             'diametter',
                                             'Heigth',
                                             'whole weight',
                                             'shucked weight',
                                             'viscera weight',
                                             'shell weight',
                                             'Rings'])
pd.plotting.scatter_matrix(dataset)
X = dataset.iloc[:,0:8].values
y = dataset.iloc[:,-1].values


dataset.isnull().sum()
#imputer is not use because no nall values
#temp = pd.DataFrame(X)
#temp[0].value_counts()
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,0] = lab.fit_transform(X[:,0])
lab.classes_



from sklearn.preprocessing import OneHotEncoder
one =   OneHotEncoder(categorical_features = [0])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X = sts.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X,y)
lr.score(X_test,y_test)
y_pred = lr.predict(X)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X,y)
y_pred1 = log.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred1)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train,y_train)
dt.score(X,y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
knn.score(X,y)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_test,y_test)
nb.score(X_test,y_test)


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)

from sklearn.ensemble import voting_classifier
vott = voting_classifier([('LR',lr),
                          ('Log',log),
                          ('SVM',svm),
                          ('NB',nb)])


















