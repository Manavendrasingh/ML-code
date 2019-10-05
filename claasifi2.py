import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split # split in to set of data 
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.5)

from sklearn.linear_model import LogisticRegression #implement Logistic Regression 
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

log_reg.score(X_train,y_train)# check score on train dataset 
log_reg.score(X_test,y_test)# check score on test dataset
log_reg.score(X,y)#check score on full dataset


y_pred = log_reg.predict(X)


from sklearn.metrics import confusion_matrix# to creat confusion metricx
cm = confusion_matrix(y,y_pred)
#cm.fit(y,y_pred)


from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y,y_pred) 
recall_score(y,y_pred)
f1_score(y,y_pred)






