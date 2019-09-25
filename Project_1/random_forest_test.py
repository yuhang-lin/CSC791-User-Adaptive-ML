#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Yuhang Lin

import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

def impute_df(imputer, output_folder, data_folder, skip=0):
    os.makedirs(output_folder, exist_ok=True)
    patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None) 
    for patient in patients.iloc[:, 0]:
        if patient < skip:
            continue
        input_df = pd.read_csv("{}/test_with_missing/{}.csv".format(data_folder, patient))
        imputed_df = pd.DataFrame(imputer.fit_transform(input_df))
        imputed_df.columns = input_df.columns
        imputed_df.index = input_df.index
        imputed_df.to_csv("{}/{}.csv".format(output_folder, patient), index=False)
        del input_df
        del imputed_df

data_folder = './test_data'
estimators = [
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=200, random_state=0),
    KNeighborsRegressor(n_neighbors=5)
]
max_iter = 40

names = ['decisiontree', 'extratrees', 'knn']
for i in [1]:
    name = names[i]
    estimator = estimators[i]
    output_folder = "./test_output/iterative_imputer_{}_iter{}".format(name, max_iter)
    imputer = IterativeImputer(max_iter=max_iter, random_state=0, estimator=estimator)
    impute_df(imputer, output_folder, data_folder)




