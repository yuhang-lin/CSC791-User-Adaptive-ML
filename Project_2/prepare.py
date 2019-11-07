#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd


def prepare(feature_list, output_folder=".", data_folder=".", file_name="Training_data"):
    # save selected features to a csv file
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv("{}/MDP_Original_data.csv".format(data_folder), dtype='str')
    NUM_STATIC = 6
    MAX_LENGTH = NUM_STATIC + 8 # up to eight features
    keep_list = [i for i in range(NUM_STATIC)]
    feature_list.sort()
    for item in feature_list:
        if item in keep_list:
            continue
        keep_list.append(item)
    if len(keep_list) == NUM_STATIC:
        return # we have no feature actually
    if len(keep_list) > MAX_LENGTH:
        print("Feature list has size over the maximum limit of {}".format(MAX_LENGTH - NUM_STATIC))
    df[df.columns[keep_list]].to_csv("{}/{}.csv".format(output_folder, file_name), index=False)


if __name__ == '__main__':
    prepare([16, 17])
