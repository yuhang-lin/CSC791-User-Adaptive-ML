#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Yuhang Lin

import os
import pandas as pd

model_dir1 = './Baseline/MICE_train_imputed'
model_dir2 = './output/iterative_imputer_extratrees_iter40'
data_dir = './train_data'
columns_1 = ["X5", "X6"] # the columns from model 1
output_dir = "./output/mice_extratrees_40"

patients = pd.read_csv("{}/pts.tr.csv".format(data_dir), header=None)
os.makedirs(output_dir, exist_ok=True)
for patient in patients.iloc[:, 0]:
    df_1 = pd.read_csv("{}/{}.csv".format(model_dir1, patient))
    df_2 = pd.read_csv("{}/{}.csv".format(model_dir2, patient))
    # replace those columns by the same columns from model 1
    for i in range(len(columns_1)):
        df_2[columns_1[i]] = df_1[columns_1[i]]
    df_2.to_csv("{}/{}.csv".format(output_dir, patient), index=False)
    del df_1
    del df_2

