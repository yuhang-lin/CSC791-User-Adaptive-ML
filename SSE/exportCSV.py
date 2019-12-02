#!/usr/bin/env python
# coding: utf-8

import csv

def exportCSV(data, fileName='output.csv'):
    """
    data is a list storing the data to be appended to the csv fileName
    """
    with open(fileName, 'a+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(data)


if __name__ == "__main__": # example for running the function
        data=[3, 4, 4.5]
        exportCSV(data, 'test.csv')
