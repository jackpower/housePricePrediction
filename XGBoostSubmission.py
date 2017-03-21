#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:38:43 2017

@author: Jack
"""
# Import libraries
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cross_validation import KFold
import datetime

# Set random state
rng = np.random.RandomState(9462)

# Set working directory
import os
os.chdir('/Users/Jack/Documents/housePrice')

# Set names of input files
train_name = 'data/train.csv'
test_name = 'data/test.csv'
submission_col = 'SalePrice'
submission_name = 'data/sample_submission.csv'

# Load data
train = pd.DataFrame.from_csv(train_name)
test = pd.DataFrame.from_csv(test_name)
submission = pd.DataFrame.from_csv(submission_name)

# Extract target variable
target = train['SalePrice']
del train['SalePrice']

## Define predictors

columns = train.columns.values  # Define the names of columns

# Another version of filling missing values where missing values 
# are simply replaced by -1, interestingly achieves superior performance than model below. 
# Next step: look into reasons why. Potentially missing value prediction can be done more effectively
# e.g. Random Forest or perhaps it is only effective on numerical variables.
#train = train.fillna(-1)
#test = test.fillna(-1)

# Change missing values in nominal variables where missing infers additional information in train and test sets
train['Alley'] = train['Alley'].fillna("No Alley")
train['FireplaceQu'] = train['FireplaceQu'].fillna("No Fireplace")
train['GarageQual'] = train['GarageQual'].fillna("No Garage")
train['GarageCond'] = train['GarageCond'].fillna("No Garage")
train['PoolQC'] = train['PoolQC'].fillna("No Pool")
train['Fence'] = train['Fence'].fillna("No Fence")
train['MiscFeature'] = train['MiscFeature'].fillna("No Feature")

test['Alley'] = test['Alley'].fillna("No Alley")
test['FireplaceQu'] = test['FireplaceQu'].fillna("No Fireplace")
test['GarageQual'] = test['GarageQual'].fillna("No Garage")
test['GarageCond'] = test['GarageCond'].fillna("No Garage")
test['PoolQC'] = test['PoolQC'].fillna("No Pool")
test['Fence'] = test['Fence'].fillna("No Fence")
test['MiscFeature'] = test['MiscFeature'].fillna("No Feature")

# Replace missing values by median or mode values for factor and numerical variables respectively
numerical_features=train.select_dtypes(include=["float","int","bool"]).columns.values
categorical_features=train.select_dtypes(include=["object"]).columns.values                                 
                                        
for feature in numerical_features: 
    train[feature] = train[feature].fillna(train[feature].median())
    test[feature] = test[feature].fillna(test[feature].median())
    
for feature in categorical_features: 
    train[feature] = train[feature].fillna(train[feature].value_counts().idxmax()) # replace by most frequent value
    test[feature] = test[feature].fillna(test[feature].value_counts().idxmax()) # replace by most frequent value

# Label nominal variables to numbers
nom_numeric_cols = ['MSSubClass'] # define nominal variables currently being read as numeric
dummy_train = []
dummy_test = []
for col in columns:
    # Only works for nominal data without a lot of factors, as otherwise too many columns produced
    if train[col].dtype.name == 'object' or col in nom_numeric_cols:
        dummy_train.append(pd.get_dummies(train[col].values.astype(str), col))
        dummy_train[-1].index = train.index
        dummy_test.append(pd.get_dummies(test[col].values.astype(str), col))
        dummy_test[-1].index = test.index
        del train[col]
        del test[col]

# Merge back into train & test datasets
train = pd.concat([train] + dummy_train, axis=1)
test = pd.concat([test] + dummy_test, axis=1)

# Turn YearBuilt into YearsSince / Old variables. Assumes that data published in 2012.
train['yearsOld'] = 2012 - train['YearBuilt']
train['yearsSinceRemodel'] = 2012 - train['YearRemodAdd']
train['yearsSinceSale'] = 2012 - train['YrSold']

test['yearsOld'] = 2012 - test['YearBuilt']
test['yearsSinceRemodel'] = 2012 - test['YearRemodAdd']
test['yearsSinceSale'] = 2012 - test['YrSold']

# Remove variables accounted for elsewhere
del train['YearBuilt']
del train['YearRemodAdd']
del train['YrSold']
del test['YearBuilt']
del test['YearRemodAdd']
del test['YrSold']

## Build model
err = []

# Turn into array so that data can be read by XGBoost
train = np.array(train)
target = np.array(target)
test = np.array(test)
print(train.shape, target.shape, test.shape)

kfold = KFold(train.shape[0], n_folds=5, random_state=rng) # Define the k fold method

# Train the initial model
for cv_train_index, cv_test_index in kfold:
    # Define the train and test datasets
    X_train, X_test = train[cv_train_index, :], train[cv_test_index, :]
    y_train, y_test = target[cv_train_index], target[cv_test_index]

    # Define dataset, put in DMatrix for XGBoost
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)

    # Define the model
    xgb_model = xgb.XGBRegressor(seed=rng,nthread=4).fit(X_train, y_train)

    # Predict results & calculate error
    predicted_results = xgb_model.predict(X_test)
    actuals = y_test    
    err.append(mean_squared_error(actuals, predicted_results))

# Predict model on one train & test dataset & calculate MSE
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=rng)
predcited = xgb_model.predict(X_test)
actuals = y_test
print(mean_squared_error(actuals,predcited))


## Hyperparameter tuning

# Round 1 - Tune number of boosting trees, initially
param_grid = {
 'n_estimators':range(50,1000,25)
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1) # Use grid search to find optimal parameter, measure with MSE

clf.fit(train,target) # Fit model
    
bestParams1 = clf.best_params_ # Set best parameters
bestScore1 = clf.best_score_ # Record best score

xgb_model.set_params(n_estimators=bestParams1['n_estimators']) # Set parameter in model

# Round 2 - Tune max depth & min chid weight

param_grid = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2),
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams2 = clf.best_params_
bestScore2 = clf.best_score_

xgb_model.set_params(max_depth=bestParams2['max_depth'],min_child_weight=bestParams2['min_child_weight'])

# Round 3 - Tune gamma

param_grid = {
 'gamma':[i/10.0 for i in range(0,5)]
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams3 = clf.best_params_
bestScore3 = clf.best_score_

xgb_model.set_params(gamma=bestParams3['gamma'])

# Round 4 - Confirm optimal number of number boosted trees with new parameters

param_grid = {
 'n_estimators':range(50,1000,25)
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams4 = clf.best_params_
bestScore4 = clf.best_score_

xgb_model.set_params(n_estimators=bestParams4['n_estimators'])

# Round 5 - Tune subsample & colsample

param_grid = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams5 = clf.best_params_
bestScore5 = clf.best_score_

xgb_model.set_params(subsample=bestParams5['subsample'],colsample_bytree=bestParams5['colsample_bytree'])

# Round 6 - Tune number of trees with lower learning rate

xgb_model.set_params(learning_rate=0.005)

param_grid = {
 'n_estimators':range(50,1000,5)
}

clf = GridSearchCV(xgb_model, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose=1)

clf.fit(train,target)
    
bestParams6 = clf.best_params_
bestScore6 = clf.best_score_

xgb_model.set_params(n_estimators=bestParams6['n_estimators'])


## Retrain model based on tuned parameters (same as above)

for cv_train_index, cv_test_index in kfold:
    X_train, X_test = train[cv_train_index, :], train[cv_test_index, :]
    y_train, y_test = target[cv_train_index], target[cv_test_index]

    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)

    xgb_model = xgb_model.fit(X_train, y_train)

    predicted_results = xgb_model.predict(X_test)
    actuals = y_test    
    err.append(mean_squared_error(actuals, predicted_results))

## Create a submission for Kaggle.

submission[submission_col] = xgb_model.predict(test)

now = datetime.datetime.now()

reference = str(now.day) + '-' + str(now.month) + '-' + str(now.year) + ' ' + str(now.hour) + ':' + str(now.minute)
x = reference
y = '.csv'

submission_target = "submissions/%s%s" % (x,y)

submission.to_csv(submission_target)