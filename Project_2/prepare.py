#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd


def prepare(feature_list, source_file, output_file="Training_data.csv"):
    df = pd.read_csv("{}".format(source_file), dtype='str')
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
        keep_list = keep_list[:MAX_LENGTH] # discard all the extra features
    df[df.columns[keep_list]].to_csv("{}".format(output_file), index=False)


if __name__ == '__main__':
    prepare([16, 17], "MDP_Original_data") # create a file similar to sample_training_data.csv
