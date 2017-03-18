#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:37:23 2017

@author: Jack
"""

#Import libraries:
import os
os.chdir('/Users/Jack/Documents/housePrice')

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('data/train.csv')
train = train.fillna(-1)
train['SalePrice'] = np.log(train['SalePrice'])
target = 'SalePrice'
IDcol = 'Id'

def modelfit(alg, dtrain, predictors, useTrainCV=False, cv_folds=5):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics = 'auc')
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
        
    #Print model report:
    print "\nModel Report"
    print "RMSE : %f" % np.sqrt(mean_squared_error(dtrain[target].values, dtrain_predictions))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    
now = datetime.datetime.now()

train['yearsOld'] = now.year - train['YearBuilt']
train['yearsSinceRemodel'] = now.year - train['YearRemodAdd']
train['yearsSinceSale'] = now.year - train['YrSold']

#test['yearsOld'] = now.year - test['YearBuilt']
#test['yearsSinceRemodel'] = now.year - test['YearRemodAdd']
#test['yearsSinceSale'] = now.year - test['YrSold']

predictors = [x for x in train.columns if x in ['LotFrontage','LotArea','OverallQual','OverallCond','yearsOld','yearsSinceRemodel','yearsSinceSale',
'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']]

xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 seed=27)

modelfit(xgb1, train, predictors)

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', seed=27), 
 param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_