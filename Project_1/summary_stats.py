import pandas as pd
import numpy as np

def summary_per_analyte(data_choice, data_type, data_folder):
    if data_type == 'train':
        if data_choice == 'G':
            data = '/train_groundtruth/'
        else:
            data = '/train_with_missing/'

    if data_type == 'test':
        data = '/test_with_missing/'

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

    f = open('{}_output.txt'.format(data.strip('/')), 'w+')
    f.write('Counts of missing value per analyte over all patients is: {}\n'.format(summation))
    f.write('Average number of missing values per analyte is: {}\n'.format(np.mean(summation)))
    f.write('SD of number of missing values per analyte is: {}\n'.format(np.std(summation)))
    f.write('Mean number of rows over all patients is: {}\n'.format(np.mean(length_list)))
    f.write('Median number of rows over all patients is: {}\n'.format(np.median(length_list)))
    f.write('SD of number of rows over all patients is: {}\n'.format(np.std(length_list)))
    f.close()



def patients_stats(data_folder):
    patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None)
    print('Number of patients: {}'.format(len(patients.iloc[:,0])))

    masked_data = pd.read_csv("{}/naidx.csv".format(data_folder), header=0)

    # Number of masked value per patient
    num_masked = masked_data.groupby(['pt.num']).size()
    print(num_masked)

    # for each patient, verify if eaxctly one value is missing per analyte
    masked_value = masked_data.groupby(['pt.num']).test.nunique()
    print(masked_value)


# Choose data choice between 'G' for groundtruth and 'M' for missing
# Choose data type between 'train' and 'test'.
# Choose data folder between './train_data' and './test_data'
# summary_per_analyte(data_choice='M', data_type='test', data_folder='./test_data')
patients_stats(data_folder='./test_data')