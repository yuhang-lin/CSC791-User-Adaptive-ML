import pandas as pd
import numpy as np

def summary_per_analyte(data_folder = './train_data'):
    patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None)
    first_patient = pd.read_csv("{}/train_groundtruth/{}.csv".format(data_folder, 1))

    #Initilize an empty list for storing average counts of missing value per analyte
    # Missing value per analyte in ground truth
    miss_val = []
    empty_list = []
    count = 0
    length_list = []            # no of rows per patient

    column_length = len(first_patient.columns)

    for i in range(column_length):
        miss_val.append(empty_list)

    for patient in patients.iloc[:, 0]:
        ground_truth_df = pd.read_csv("{}/train_groundtruth/{}.csv".format(data_folder, patient))

        for j in range(column_length):
            count = ground_truth_df.iloc[:, j].isnull().sum()   # Count
            miss_val[j].append(count)

        length_list.append(ground_truth_df.shape[0])        # appending to list of number of rows
    print('Mean number of rows is: {}'.format(np.mean(length_list)))
    print('Median number of rows is: {}'.format(np.median(length_list)))
    print('SD of number of rows is: {}'.format(np.std(length_list)))



def patients_stats(data_folder = './train_data'):
    patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None)
    print('Number of patients: {}'.format(len(patients.iloc[:,0])))

    masked_data = pd.read_csv("{}/naidx.csv".format(data_folder), header=0)

    # Number of masked value per patient
    num_masked = masked_data.groupby(['pt.num']).size()
    print(num_masked)

    # for each patient, verify if eaxctly one value is missing per analyte
    masked_value = masked_data.groupby(['pt.num']).test.nunique()
    print(masked_value)



summary_per_analyte()