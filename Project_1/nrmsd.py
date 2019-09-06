#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Yuhang Lin

import pandas as pd
import math

def getNRMSD(data_folder="./train_data", evaluate_folder="./Baseline/MICE_train_imputed"):
    masks = pd.read_csv("{}/naidx.csv".format(data_folder))

    last_patient = 0
    predicted_df = None
    truth_df = None
    NUM_TEST = 13
    nRMSD = dict()
    for i in range(1, NUM_TEST+1):
        nRMSD["X{}".format(i)] = []
    for i, row in masks.iterrows():
        patient = row['pt.num']
        test = row["test"]
        row_num = row['i']
        if patient != last_patient:
            del predicted_df
            del truth_df
            predicted_df = pd.read_csv("{}/{}.csv".format(evaluate_folder, patient))
            truth_df = pd.read_csv("{}/train_groundtruth/{}.csv".format(data_folder, patient))
            last_patient = patient
        predict_val = predicted_df.iloc[row_num-1][test]
        truth_val = truth_df.iloc[row_num-1][test]
        truth_test_range = truth_df[test].max() - truth_df[test].min()
        nRMSD[test].append(((predict_val - truth_val)/truth_test_range) ** 2)

    for key in nRMSD:
        values = nRMSD[key]
        res = math.sqrt(sum(values)/len(values))
        print("{:.7f}".format(res))

if __name__== "__main__":
    getNRMSD()
