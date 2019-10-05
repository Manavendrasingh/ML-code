import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m=100 
X=6*np.random.randn(m,1)-3# to generate random feature metrix
y=0.5*X**2+X+2+np.random.randn(m,1)# generate rendom vector of prediction

plt.scatter(X,y)
plt.axis([-3,3,0,9])#to zoom the graph[Xaxis(range),yaxis(range)]
plt.show()

from sklearn.preprocessing  import PolynomialFeatures
ply=PolynomialFeatures(degree=2,include_bias=False)
X_pl=ply.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X_pl,y)
 #creat testing data 
X_new=np.linspace(-3,3,100).reshape(-1,1)#
X_new_ply=ply.fit_transform(X_new)
y_new=lin.predict(X_new_ply)

plt.scatter(X,y)
plt.plot(X_new,y_new,c="r")
plt.axis([-3,3,0,9])
plt.show()

lin.coef_
lin.intercept_











