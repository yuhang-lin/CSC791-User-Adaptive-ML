import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import csv

def normalize_df(df_org):
    # the time column
    tim = pd.DataFrame(df_org['time'])

    # the class column
    clas = pd.DataFrame(df_org['class'])

    # drop time from the dataset so that we don't scale the timestamps
    df = df_org.drop(['time', 'class'], axis=1)

    # Get the headers from the list
    headers = list(df)

    scaler = StandardScaler()
    scaled_values = pd.DataFrame(scaler.fit_transform(df), columns = headers, index = df.index)
    
    # concatenate the data after scaling it
    df_new = pd.concat([tim, clas, scaled_values], axis=1)
    df_new = df_new.astype({'class': 'int32'})
    return df_new

def get_square_df(df):
    new_df = df.pow(2)
    # the time column
    tim = pd.DataFrame(df['time'])

    # the class column
    clas = pd.DataFrame(df['class'])

    # drop time from the dataset so that we don't scale the timestamps
    new_df.drop(['time', 'class'], axis=1, inplace=True)

    # concatenate the data after scaling it
    new_df = pd.concat([tim, clas, new_df], axis=1)
    new_df = new_df.astype({'class': 'int32'})
    return new_df

def RMS(df, start, end):
    return df.iloc[start:end + 1, :].mean()

def partition_window(df, window_size=200, step_size=100):
    count = 0
    rows, columns = df.shape
    last_data = []
    start = 0
    end = 0
    square_df = get_square_df(df)
    RMS_list = []
    window_list = []
    
    for i in range(0, rows, step_size):
        if end > 0:
            start = end - step_size + 1
        else:
            start = i
        end = min(i + window_size - 1, rows - 1)
        #count += 1 # count of total possible windows we can have
        #print("{} to {}".format(start, end))
        res = RMS(square_df, start, end)
        if res[1] == int(res[1]) and int(res[1]) > 0 and int(res[1]) < 7 and end - start + 1 == window_size:
            RMS_list.append(res)
            window_list.append([start, end])
        if end - start + 1 < window_size:
            count += 1
        if end == rows - 1:
            break
    return window_list, RMS_list, count

def exportCSV(data, fileName='output.csv'):
    """
    data is a list storing the data to be appended to the csv fileName
    """
    with open(fileName, 'a+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(data)

# train, test split
def train_valid_test_split(path = './EMG_data_for_gestures-master/'):
    # path = './drive/My Drive/EMG_data_for_gestures-master/'
    test_file = 1 # 1 for using the first half of the second text file
    num_subject = 36 # number of subjects we have 
    # find all the changing points first, 
    # then use the first changing point from 6 to 0 minus 1 to be the row to split.
    # split the data into train and test, then return list of dataframes
    # a list holding the data for all 36 users
    log_file = path + '{}.csv'.format("change_points")
    change_points = [[] for i in range(num_subject)]
    is_loaded = False
    if os.path.isfile(log_file):
        # load the data
        count = 0
        with open(log_file, 'r') as fin:
            for line in fin:
                count += 1
                if count % 2 == 1:
                    continue
                data = list(map(int, line.strip().split(",")))
                change_points[(count - 1) // 4].append(data)
        is_loaded = True
        # for i in range(num_subject):
        #     print(change_points[i])

    training = []
    validation = []
    testing = []
    for i in range(num_subject):  # since there are 36 users
        all_files = glob.glob(path + "{}/*.txt".format(i+1))
        # read both files for each user
        for j, filename in enumerate(all_files):
            df = pd.read_csv(filename, sep = "\t")
            if not is_loaded:
                change_points[i].append([])
                last_class = 0
                classes = []
                for index, row in enumerate(df.itertuples(index=False)):
                    if row[-1] != last_class:
                        last_class = row[-1]
                        classes.append(last_class)
                        change_points[i][j].append(index)
                #print(classes)
                exportCSV(classes, log_file)
                #print(change_points[i][j])
                exportCSV(change_points[i][j], log_file)
            if j == test_file:
                #print("{}, {}".format(i, j))
                #print(change_points[i][j])
                split_point = change_points[i][j][11]
                # append first half to validation,
                # second half to testing
                validation.append(df.iloc[:split_point,:])
                testing.append(df.iloc[split_point:,:])
            else:
                # append this whole df to training
                training.append(df)
    return training, validation, testing

def getXY(df):
    # normalize this df
    df = normalize_df(df)
    # partition this df into windows
    window_list, RMS_list, count = partition_window(df)
    # get the X and y from RMS list
    rmsX = [list(i[2:]) for i in RMS_list]
    dataX = [df.iloc[window[0]:window[1] + 1,2:] for window in window_list]
    y = [int(i[1]) - 1 for i in RMS_list] # map class from [1, 6] to [0, 5] for softmax activation
    return rmsX, dataX, y
