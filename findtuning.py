import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y =dataset.target

from sklearn.linear_model import LinearRegression
log_reg = LinearRegression()


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


from sklearn.svm import SVC
svm = SVC()

from sklearn.naive_bayes import GaussianNB
nb=  GaussianNB()

#voting ..........
from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('LR',log_reg),
                        ('KNN',knn),
                        ('DT',dt),
                        ('NB',nb),
                        ('SVM',svm)])
vot.fit(X,y)
vot.score(X,y)




#bagging.............
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(nb, n_estimators = 5)
bag.fit(X,y)
bag.score(X,y)



#random forest classifires
from sklearn.ensemble import RandomForestClassifier
ran = RandomForestClassifier(n_estimators = 5)
ran.fit(X,y)
ran.score(X,y)







  

