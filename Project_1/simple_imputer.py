#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Yuhang Lin

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from base import impute_df

strategy = 'mean'
data_folder = './train_data'
output_folder = "./output/simple_imputer_{}".format(strategy)

imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
impute_df(imputer, output_folder, data_folder)
