import glob
import pandas as pd

def read_user_data():
    """
    A function that reads all the user data and prepares them for future use
    :return: a list of all the user data
    """
    
    # path of the dataset
    path = './EMG_data_for_gestures-master/'  
    
    # a list holding the data for all 36 users
    user_data = []

    for i in range(36):  # since there are 36 users
        all_files = glob.glob(path + "{}/*.txt".format(i+1))
        
        # two files for each user, hence two dataframes
        file = [pd.DataFrame() for _ in range(2)]
        
        # read both files for each user
        for j, filename in enumerate(all_files):
            file[j] = pd.read_csv(filename, sep = "\t")
            
        merged_df = pd.concat([file[0], file[1]], axis=0, ignore_index=True)
        merged_df = merged_df.sort_values(by=['time'])

        # TIMING INFO (will move to a different timing function if needed)
        # insert new column of time intervals after sorting
        subtract_operand = merged_df['time'].shift(1)
        subtract_operand.at[0] = merged_df.at[0, 'time']
        intervals = merged_df['time'] - subtract_operand
        merged_df.insert(1, 'time_intervals', intervals)
        #
        
        user_data.append(merged_df)

    return user_data


def get_summary_stats(user_data):
    """
    A function that computes the summary statistics of the list of user_data.
    Print a file "summary_stats.txt" containing summary stats for all users.
    :param user_data: A list of all the user data
    :return:
    """

    columns = ["mean", "SD", "min", "max"]  # columns = ["mean", "SD", "min", "max", "median", "mode"]
    
    summary_stats = []
    for user in user_data:
        means = user.mean(axis=0).to_frame()
        std = user.std(axis=0).to_frame()
        mini = user.min(axis=0).to_frame()
        maxm = user.max(axis=0).to_frame()
        result = pd.concat([means, std, mini, maxm], axis=1, ignore_index=True)
        result.columns = columns
        summary_stats.append(result)

    f = open("summary_stats.txt", "w+")
    f.write("Summary stats for all users\n\n")

    for i, summary in enumerate(summary_stats):
        f.write("---- user {} ----\n".format(i+1))
        f.write(str(summary))
        f.write("\n\n")

    f.close()


def get_class_distribution(user_data):
    """
    A function that prints the class distribution of the user data
    Prints a file "class_distribution.txt" containing summary stats for all users
    :param user_data: A list of all the user data
    :return:
    """
    counts = []

    for user in user_data:
        count = pd.DataFrame(user['class'].value_counts())
        counts.append(count)

    return counts


'''
def get_timing_stats(user_data):
    """
    (If needed) A function that gathers timing specific stats for the data
    :param user_data: A list of all the user data
    :return:
    """
    
   
'''

"""
Use the following code to run the files
"""

# get all the user data
user_data = read_user_data()

get_summary_stats(user_data)

class_dis = get_class_distribution(user_data)
print(class_dis)








