#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd


def prepare(feature_list, output_folder=".", data_folder="."):
    """
    Prepare a CSV file "Training_data.csv" with static information along with selected features.
    """
    output_folder="."
    data_folder="."
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv("{}/MDP_Original_data.csv".format(data_folder), dtype='str')
    keep_list = [i for i in range(6)] # Keep columns 1-6 untouched
    keep_list.extend(feature_list) # Add all selected features
    df[df.columns[keep_list]].to_csv("{}/Training_data.csv".format(output_folder), index=False)


if __name__ == '__main__':
    prepare([16, 17])
