import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
dataset2 =pd.read_csv('test.csv') 


X_train = dataset.iloc[:,[0, 4, 5, 6, 7, 11]].values
y_train = dataset.iloc[:,1].values

X_test = dataset2.iloc[:,[0, 3, 4, 5, 6, 10]].values

dataset.isnull().sum()
#preprocessing of training dataset
from sklearn.preprocessing import Imputer 
imp = Imputer()
X_train[:,[0,2,3,4]] = imp.fit_transform(X_train[:,[0,2,3,4]])

temp = pd.DataFrame(X_train[:,[4,5]])
temp[1].value_counts()
temp[1] = temp[1].fillna('S')
X_train[:,[4,5]] = temp
del(temp)

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X_train[:,1] = lab.fit_transform(X_train[:,1])
lab.classes_

X_train[:,5] = lab.fit_transform(X_train[:,5].astype(str))
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one =  OneHotEncoder(categorical_features = [1,5])
X_train = one.fit_transform(X_train)
X_train = X_train.toarray()


from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X_train = sts.fit_transform(X_train)

#preprocessing of test data 
X_test[:,[0,2,3,4]] = imp.fit_transform(X_test[:,[0,2,3,4]])#imputer
X_test[:,1] = lab.fit_transform(X_test[:,1])#label Encoding
lab.classes_
X_test[:,5] = lab.fit_transform(X_test[:,5])#label Encoding
lab.classes_
one2 = OneHotEncoder(categorical_features = [1,5])#one hot encoing
X_test = one2.fit_transform(X_test)
X_test = X_test.toarray()
#feature scaling
X_test = sts.fit_transform(X_test)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 6)
dt.fit(X_train,y_train)
dt.score(X_train,y_train)
y_pred = dt.predict(X_test)
# giveing columns name  to y_pred
y_pred = y_pred.reshape(-1,1)
df = pd.DataFrame(y_pred)
df.columns = ['Survived']
y_pred = df


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.svm import SVC
svm = SVC()

from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('Log',log),('DT',dt),('KNN',knn),('NB',nb),('SVM',svm)])
vot.fit(X_train,y_train)
vot.score(X_train,y_train)
y_pred2 = vot.predict(X_test)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(nb,n_estimators=5)
bag.fit(X_train,y_train)
bag.score(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
ran = RandomForestClassifier(n_estimators=5)
ran.fit(X_train,y_train)
ran.score(X_train,y_train)
#this is complete












