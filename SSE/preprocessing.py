import glob
import pandas as pd

def read_user_data():
    """
    A function that reads all the user data and prepares them for future use
    :return:
    """

    path = './EMG_data_for_gestures-master/'  # use your path

    li = []

    for i in range(1, 37):  # since there are 37 users
        all_files = glob.glob(path + "{}/*.txt".format(i))
        print(all_files)

        for filename in all_files:
            df = pd.read_csv(filename, sep = "\t")
            li.append(df)

            frame = pd.concat(li, axis=0, ignore_index=True)
            print(frame)
        frame = None



    # df = pd.read_csv("./EMG_data_for_gestures-master/{}/1_raw_data_13-12_22.03.16.txt".format(1), sep="\t")
    # print(df)

print(read_user_data())


