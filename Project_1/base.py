import os
import pandas as pd

def impute_df(imputer, output_folder, data_folder):
    os.makedirs(output_folder, exist_ok=True)
    patients = pd.read_csv("{}/pts.tr.csv".format(data_folder), header=None) 
    for patient in patients.iloc[:, 0]:
        input_df = pd.read_csv("{}/train_with_missing/{}.csv".format(data_folder, patient))
        imputed_df = pd.DataFrame(imputer.fit_transform(input_df))
        imputed_df.columns = input_df.columns
        imputed_df.index = input_df.index
        imputed_df.to_csv("{}/{}.csv".format(output_folder, patient), index=False)
        del input_df
        del imputed_df