import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
processed_reviews = [] 

dataset = pd.read_csv('At&T_Data.csv')
y = dataset.iloc[:,-1].values
#preproce3ssing of cloumn 
from sklearn.preprocessing  import LabelEncoder
lab = LabelEncoder()
y = lab.fit_transform(y)
y = y.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [0])
y = one.fit_transform(y)
y = y.toarray()

from sklearn.model_selection import train_test_split
y_train,y_test = train_test_split(y)

#preprocessing of reviews column
for i in range(113):
    Reviews = re.sub('[^a-zA-Z]',' ',dataset['Reviews'][i])# this line remove all imoges and symbols
    Reviews = Reviews.lower()
    Reviews = Reviews.split()    
    Reviews = [ps.stem(tok) for tok in Reviews if  not tok in set(stopwords.words(['english'])) ]#remove all helping verb and articals,ing ,es,s etc in last,
    #in this stopword.word cantian words like ing,the  etc which is not reguired  and stem remove es and ing etc from word which require for processing 
    Reviews = ' '.join(Reviews)#join the seprate word again in reviews
    processed_reviews.append(Reviews)
 
from sklearn.feature_extraction.text import CountVectorizer #this is used to find 100 word which is    
cv = CountVectorizer(max_features = 100)
X = cv.fit_transform(processed_reviews)
X = X.toarray()
print(X)
print(cv.get_feature_names())#shiw the words which are selected 
print(cv.vocabulary_)#show the all among the words  vocavulary
from sklearn.model_selection import train_test_split
X_train,X_test = train_test_split(X)

process_title = []
for i in range(113):
    Titles = re.sub('[^a-zA-z]',' ',dataset['Titles'][i])
    Titles = Titles.lower()
    Titles  = Titles.split()
    Titles = [ps.stem(tite) for tite in Titles if not tite in set(stopwords.words(['english']))]
    Titles = ' '.join(Titles)
    process_title.append(Titles)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)
Y = cv.fit_transform(process_title )
Y = Y.toarray()
print(cv.get_feature_names())
print(cv.vocabulary_)
from sklearn.model_selection import train_test_split
Y_train,Y_test = train_test_split(Y)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train,y_train)
dt.score(X_train,y_train)
dt.score(X_test,y_test)

dt.fit(Y_train,y_train)
dt.score(Y_test,y_test)
dt.score(Y_train,y_train)


    
    