#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:38:43 2017

@author: Jack
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
import datetime

rng = np.random.RandomState(9462)

import os
os.chdir('/Users/Jack/Documents/housePrice')

# Load data
train_name = 'data/train.csv'
test_name = 'data/test.csv'
submission_col = 'SalePrice'
submission_name = 'data/sample_submission.csv'

now = datetime.datetime.now()

reference = str(now.day) + '-' + str(now.month) + '-' + str(now.year) + ' ' + str(now.hour) + ':' + str(now.minute)
x = reference
y = '.csv'

submission_target = "submissions/%s%s" % (x,y)

# Read files
train = pd.DataFrame.from_csv(train_name)
test = pd.DataFrame.from_csv(test_name)
submission = pd.DataFrame.from_csv(submission_name)

# Fill na's
train = train.fillna(-1)
test = test.fillna(-1)

# Extract target
target = train['SalePrice']
del train['SalePrice']

columns = train.columns.values

# Define predictors
# Label nominal variables to numbers
nom_numeric_cols = ['MSSubClass'] # nominal variables being read as numeric
dummy_train = []
dummy_test = []
for col in columns:
    # Only works for nominal data without a lot of factors
    if train[col].dtype.name == 'object' or col in nom_numeric_cols:
        dummy_train.append(pd.get_dummies(train[col].values.astype(str), col))
        dummy_train[-1].index = train.index
        dummy_test.append(pd.get_dummies(test[col].values.astype(str), col))
        dummy_test[-1].index = test.index
        del train[col]
        del test[col]

train = pd.concat([train] + dummy_train, axis=1)
test = pd.concat([test] + dummy_test, axis=1)

train['yearsOld'] = 2012 - train['YearBuilt']
train['yearsSinceRemodel'] = 2012 - train['YearRemodAdd']
train['yearsSinceSale'] = 2012 - train['YrSold']

test['yearsOld'] = 2012 - test['YearBuilt']
test['yearsSinceRemodel'] = 2012 - test['YearRemodAdd']
test['yearsSinceSale'] = 2012 - test['YrSold']

# Define CV
err = []

train = np.array(train)
target = np.array(target) # Chang to log
test = np.array(test)
print(train.shape, target.shape, test.shape)

kfold = KFold(train.shape[0], n_folds=5, random_state=rng)

for cv_train_index, cv_test_index in kfold:
    X_train, X_test = train[cv_train_index, :], train[cv_test_index, :]
    y_train, y_test = target[cv_train_index], target[cv_test_index]

    # train machine learning
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)

    xgb_model = xgb.XGBRegressor(seed=rng,nthread=4).fit(X_train, y_train)

    # predict
    predicted_results = xgb_model.predict(X_test)
    actuals = y_test    
    err.append(mean_squared_error(actuals, predicted_results))

## Round 1
param_grid = {
 'n_estimators':range(50,500,25)
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams1 = clf.best_params_
bestScore1 = clf.best_score_

xgb_model.set_params(n_estimators=bestParams1['n_estimators'])

## ROUND 2

param_grid = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2),
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams2 = clf.best_params_
bestScore2 = clf.best_score_

xgb_model.set_params(max_depth=bestParams2['max_depth'],min_child_weight=bestParams2['min_child_weight'])

## ROUND 3

param_grid = {
 'gamma':[i/10.0 for i in range(0,5)]
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams3 = clf.best_params_
bestScore3 = clf.best_score_

xgb_model.set_params(gamma=bestParams3['gamma'])

## ROUND 4

xgb_model = xgb.XGBRegressor(seed=rng,n_estimators=200,max_depth=5,min_child_weight=3,gamma=0)

param_grid = {
 'n_estimators':range(50,500,25)
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams4 = clf.best_params_
bestScore4 = clf.best_score_

xgb_model.set_params(n_estimators=bestParams4['n_estimators'])

## ROUND 5

param_grid = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams5 = clf.best_params_
bestScore5 = clf.best_score_

xgb_model.set_params(subsample=bestParams5['subsample'],colsample_bytree=bestParams5['colsample_bytree'])

## ROUND 6

xgb_model.set_params(learning_rate=0.01)

for cv_train_index, cv_test_index in kfold:
    X_train, X_test = train[cv_train_index, :], train[cv_test_index, :]
    y_train, y_test = target[cv_train_index], target[cv_test_index]

    # train machine learning
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)

    xgb_model = xgb_model.fit(X_train, y_train)

    # predict
    predicted_results = xgb_model.predict(X_test)
    actuals = y_test    
    err.append(mean_squared_error(actuals, predicted_results))

submission[submission_col] = xgb_model.predict(test)
submission.to_csv(submission_target)