#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Yuhang Lin

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from base import impute_df

data_folder = './train_data'
estimators = [
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=30, random_state=0),
    KNeighborsRegressor(n_neighbors=5)
]
max_iter = 10

names = ['decisiontree', 'extratrees', 'knn']
for i in range(len(estimators)):
    name = names[i]
    estimator = estimators[i]
    output_folder = "./output/iterative_imputer_{}_iter{}".format(name, max_iter)
    imputer = IterativeImputer(max_iter=max_iter, random_state=0, estimator=estimator)
    impute_df(imputer, output_folder, data_folder)