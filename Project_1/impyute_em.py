#!/usr/bin/env python
# coding: utf-8

# 1

import sys
get_ipython().system('pip3 install impyute')
import numpy as np
import pandas as pd
import impyute as impy
import os

# 2

strategy = 'em'
iterations = 10
data_folder = './train_data'
output_folder = "./output/impyute_{}_loops{}".format(strategy, iterations)
try:
    os.makedirs(output_folder)
except FileExistsError:
    # directory already exists
    pass
patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None)

# 3

for patient in patients.iloc[:, 0]:
    input_df = pd.read_csv("{}/train_with_missing/{}.csv".format(data_folder, patient))
    # impyute takes and returns numpy arrays: input_df.values
    imputed_df = pd.DataFrame(impy.imputation.cs.em(input_df.values, loops=iterations))
    # use existing heading names
    imputed_df.columns = input_df.columns
    imputed_df.index = input_df.index
    # output
    imputed_df.to_csv("{}/{}.csv".format(output_folder, patient), index=False)
    del input_df
    del imputed_df

#




