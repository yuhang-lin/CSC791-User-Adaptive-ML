#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Yuhang Lin

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

data_folder = './train_data'
output_folder = "./output/iterative_imputer"

try:
    os.makedirs(output_folder)
except FileExistsError:
    # directory already exists
    pass

patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None)

for patient in patients.iloc[:, 0]:
    input_df = pd.read_csv("{}/train_with_missing/{}.csv".format(data_folder, patient))
    fill_NaN = IterativeImputer(max_iter=20, random_state=0)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(input_df))
    imputed_df.columns = input_df.columns
    imputed_df.index = input_df.index
    imputed_df.to_csv("{}/{}.csv".format(output_folder, patient), index=False)
    del input_df
    del imputed_df
