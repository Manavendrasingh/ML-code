import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd
import nltk

from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import re
process_list = []

dataset = pd.read_csv('train.csv')

for i in range(31962):
    tweet = re.sub('@user',' ',dataset['tweet'][i])
    tweet = re.sub('[^a-zA-Z]',' ',dataset['tweet'][i])
    tweet = tweet.lower() 
    tweet = tweet.split() 
    tweet = [ps.stem(token) for token in tweet if not token in set(stopwords.words('english'))]                
    tweet = ' '.join(tweet)
    process_list.append(tweet)

 nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(process_list)
X = X.toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb.score(X_test,y_test)

