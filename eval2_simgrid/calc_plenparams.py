# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:20:59 2020

@author: ryuut
"""


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import datasets

copper = [(0.5, 136), 
          (1.0, 148), 
          (1.5, 160), 
          (2.0, 182), 
          (2.5, 196), 
          (3.0, 242) 
          ]

fiber = [(3.0, 1010), 
         (5.0, 1015), 
         (10 , 1070), 
         (15 , 1080), 
         (20 , 1090), 
         (30 , 1095), 
         (50 , 1780), 
         (100, 2495) 
         ]

# sns.set()

# #download datasets
# wine_data = datasets.load_wine()
# wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
# clf = linear_model.LinearRegression()
# X=wine.loc[:,['total_phenols']].values
# Y=wine['flavanoids'].values
# print(X, Y)
bw_gb = 200

for params in [copper, fiber]:
    X, Y = list(), list()
    for x, y in params:
        X.append(x)
        Y.append(y)
    
    X, Y = np.array(X), np.array(Y)
    X = X.reshape(-1, 1)
    Y = Y / bw_gb
    print(X, Y)
    clf = linear_model.LinearRegression()
    clf.fit(X,Y)
    print("回帰係数=", clf.coef_)
    print("切片：",clf.intercept_)
    print("R^2=",clf.score(X, Y))
