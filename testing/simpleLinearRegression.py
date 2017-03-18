#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:39:23 2016

@author: Jack
"""

import pandas as pd

train = pd.read_csv('/Users/Jack/Documents/housePrice/train.csv')

import statsmodels.api as sm

##TRAIN
train['yearsOld'] = 2016 - train['YearBuilt']

predictors = train[['OverallQual', 'OverallCond', 'yearsOld']]
target = train[['SalePrice']]

predictors = sm.add_constant(predictors)

est = sm.OLS(target, predictors).fit()
est.summary()


##TEST
test = pd.read_csv('/Users/Jack/Documents/housePrice/test.csv')

test['yearsOld'] = 2016 - test['YearBuilt']

new_predictors = test[['OverallQual', 'OverallCond', 'yearsOld']]
new_predictors = sm.add_constant(new_predictors)
 
test['SalePrice'] = est.predict(new_predictors)

submission = test[['Id','SalePrice']]

row_index = submission.SalePrice < 0
# then with the form .loc[row_indexer,col_indexer]
submission.loc[row_index, 'SalePrice'] = 0

submission.describe()
submission.isnull().sum()

submission.to_csv('/Users/Jack/Documents/housePrice/submission.csv',index=0)