import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('blood.xlsx')
pd.plotting.scatter_matrix(dataset)

X = dataset.iloc[2:30,1].values
y = dataset.iloc[2:30,-1].values
X = X.reshape(-1,1)

dataset.isnull().sum()

plt.scatter(X,y)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


from sklearn.linear_model import LinearRegression
lr =LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

from sklearn.ensemble import RandomForestRegressor
rans = RandomForestRegressor(n_estimators = 5)
rans.fit(X_train,y_train)
rans.score(X_test,y_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 5)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()


from sklearn.svm import SVC
svm = SVC()


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


from sklearn.neighbors import  KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)



from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('Log',log),('DT',dt),('KNN',knn),('NB',nb),('SVM',svm),('LR',lr)])
vot.fit(X_train,y_train)
vot.score(X_test,y_test)
#this is compete


