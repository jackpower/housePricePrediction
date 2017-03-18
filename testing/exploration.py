#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 13:33:41 2016

@author: Jack
"""

import pandas as pd
import os as os
os.chdir('/Users/Jack/Documents/housePrice')

import numpy
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

#turn categorical into dummies and insert mean as NA
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

import matplotlib.pyplot as plt

#Target variable description
plt.hist(train['SalePrice'], 25)
plt.show()

train['SalePrice'].describe()

correlation = train.corr()["SalePrice"]

correlation = correlation.sort_values(ascending=0)

