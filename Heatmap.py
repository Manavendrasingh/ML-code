import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('housing.csv')
pd.plotting.scatter_matrix(dataset)


corr_mat = dataset.corr()#????????????????????
import seaborn as sns
sns.heatmap(corr_mat,annot = True)