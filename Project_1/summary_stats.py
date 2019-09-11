import pandas as pd
import numpy as np

# To switch between ground truth and missing data, please change the 'data' parameter to the function below
def summary_per_analyte(data_folder = './train_data', data = '/train_groundtruth/'):
    patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None)
    first_patient = pd.read_csv("{}{}{}.csv".format(data_folder, data, 1))

    # no of rows per patient
    length_list = []

    column_length = len(first_patient.columns)

    # Iterate through every patient and count the number of rows
    for i, patient in enumerate(patients.iloc[:, 0]):
        df = pd.read_csv("{}{}{}.csv".format(data_folder, data, patient))
        length_list.append(df.shape[0])  # appending to list of number of rows

    # Initialize a 2D array with zeroes which holds count of missing values per patient per analyte
    count = np.zeros((len(length_list), column_length))

    # Initialize a 1D array with zeroes which holds count of missing values per analyte
    summation = np.zeros(column_length)

    for j in range(column_length):
        for i, patient in enumerate(patients.iloc[:, 0]):
            df = pd.read_csv("{}{}{}.csv".format(data_folder, data, patient))
            count[i][j] = df.iloc[:, j].isnull().sum()  # Count
            summation[j] += count[i][j]

    print('Counts of missing value per analyte over all patients is: {}'.format(summation))
    print('Average number of missing values per analyte is: {}'.format(np.mean(summation)))
    print('SD of number of missing values per analyte is: {}'.format(np.std(summation)))
    print('Mean number of rows over all patients is: {}'.format(np.mean(length_list)))
    print('Median number of rows over all patients is: {}'.format(np.median(length_list)))
    print('SD of number of rows over all patients is: {}'.format(np.std(length_list)))



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



patients_stats()