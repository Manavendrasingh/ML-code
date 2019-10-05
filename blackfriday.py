import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#not comlete
dataset2 = pd.read_csv('BlackFriday.csv')
dataset = dataset2.iloc[0:5000,:]


corr = dataset.corr()
corr["Purchase"].sort_values()
del(corr)

X = dataset.iloc[:,0:11].values
y = dataset.iloc[:,-1].values




dataset.isnull().sum()

temp = pd.DataFrame(X[:,[9,10]])
temp[0].value_counts()
temp[0] = temp[0].fillna("8.0")
temp[0].isnull().sum()

temp[1].value_counts()
temp[1] = temp[1].fillna("16.0")
X[:,[9,10]] = temp
del(temp)



dataset.info()
temp = pd.DataFrame(X[:,[3,6]])

temp[0].unique()
temp_age = pd.get_dummies(temp[0])
temp_age.columns = ['Age 0-17','Age18-25','Age26-35','Age36-45','Age46-50','Age51-55','Age55+']
temp = pd.concat([temp,temp_age],axis = 1)
temp.drop([0],axis = "columns",inplace = True)

temp[1].unique()
temp_age2 = pd.get_dummies(temp[1])
temp_age2.columns = ['city 0','city 1','city 2','city 3','city 4+']
temp = pd.concat([temp,temp_age2],axis = 1)
temp.drop([1],axis = "columns",inplace = True)
#
X = np.append(X,temp,axis = 1)
temp_X = pd.DataFrame(X[:,:])
#X = np.delete(X,0,axis = 1)
#X = np.delete(X,0,axis = 1)
X = np.delete(X,3,axis = 1)
X = np.delete(X,5,axis = 1)
#del(temp)




    


dataset.isnull().sum()



from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,0] = lab.fit_transform(X[:,0].astype(str))
X[:,1] = lab.fit_transform(X[:,1].astype(str))
X[:,2] = lab.fit_transform(X[:,2])
X[:,4] = lab.fit_transform(X[:,4])
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [0,1,2,4])
X  = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import Ridge
lin = Ridge(alpha = 1000)
lin.fit(X_train,y_train)
lin.score(X_test,y_test)

from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train,y_train)
svr.score(X_test,y_test)
 



from sklearn.ensemble import RandomForestRegressor
ran = RandomForestRegressor(n_estimators = 10,random_state = 2)
ran.fit(X_train,y_train)
ran.score(X_test,y_test)


from sklearn.ensemble import GradientBoostingRegressor
gr = GradientBoostingRegressor(learning_rate = 0.1,max_depth = 3,random_state = 0)
gr.fit(X_train,y_train)
gr.score(X_test,y_test)
#this is complete

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(solver = 'lbfgs')
mlp.fit(X_train,y_train)
mlp.score(X_test,y_test)

















