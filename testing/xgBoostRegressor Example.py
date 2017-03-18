#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:27:21 2017

@author: Jack
"""

import xgboost as xgb

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston

rng = np.random.RandomState(31337)

print("Boston Housing: regression")
boston = load_boston()
y = boston['target']
X = boston['data']
kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)

for train_index, test_index in kf:
    xgb_model = xgb.XGBRegressor().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(mean_squared_error(actuals, predictions))

print("Parameter optimization")
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()

param_grid = [
              {'silent': [1],
               'nthread': [2],
               'learning_rate': [0.03], #Learning Rate
               'objective': ['reg:linear'],
               'max_depth': [5, 7],
               'n_estimators': [1000],
               'subsample': [0.2, 0.4, 0.6],
               'colsample_bytree': [0.3, 0.5, 0.7],
               }
              ]

#clf = GridSearchCV(xgb_model,
#                   {'max_depth': [2,4,6],
#                    'n_estimators': [50,100,200]}, verbose=1)

clf = GridSearchCV(xgb_model, param_grid)

clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)