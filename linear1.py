import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_excel('blood.xlsx')
X=dataset.iloc[2:,1].values
y=dataset.iloc[2:,2].values

X=X.reshape(-1,1)#reshape the  vector formed a metrix 

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression#to implement LR
lin=  LinearRegression()
lin.fit(X,y)
lin.score(X,y)

plt.scatter(X,y)
plt.plot(X,lin.predict(X),c="r")#ploting line 
plt.show()

lin.predict([[500]])#give input here in place of 25 and get output
lin.coef_
lin.intercept_

 