import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

dataset = pd.read_csv('DemographicData.csv')
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,-1].values

pd.plotting.scatter_matrix(dataset)

X = np.delete(X,1,axis = 1)

dataset.isnull().sum()

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,0] = lab.fit_transform(X[:,0])
y = lab.fit_transform(y)


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [0])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X = st.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X_test,y_test)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nb.score(X_test,y_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train,y_train)
dt.score(X_test,y_test)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train,y_train)
mlp.score(X_test,y_test)

from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('Log',log),('SVM',svm),('DT',dt),('MLP',mlp)])
vot.fit(X_train,y_train)
vot.score(X_test,y_test)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(dt,n_estimators = 10)
bag.fit(X_train,y_train)
bag.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier
rand = RandomForestClassifier(n_estimators = 5)
rand.fit(X_train,y_train)
rand.score(X_test,y_test)
#this is complete

