import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#classification problem
dataset = pd.read_csv('heart.csv')
pd.plotting.scatter_matrix(dataset)
dataset.isnull().sum()
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X = sts.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X_test,y_test)
y_pred = log.predict(X_test)

from sklearn.metrics import  confusion_matrix
cm = confusion_matrix(y_test,y_pred)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 5)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.svm import SVC
svm = SVC()

from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('LR',log),
                        ('KNN',knn),
                        ('DT',dt),
                        ('NB',nb),
                        ('SVM',svm)])
vot.fit(X_train,y_train)
vot.score(X_test,y_test)

