#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system(' pip install hmmlearn')


# In[3]:

import numpy as np
import pandas as pd
from preprocessEMG import train_valid_test_split, getXY
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from statistics import mean

from HMMlearn import evaluate_hmm_model, group_training_by_class, train_hmm_models_per_user, test_hmm_models_per_user
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.neural_network import MLPClassifier

from exportCSV import exportCSV

# In[2]:


training, validation, testing = train_valid_test_split()


# In[3]:


def evaluate_user_hmms():
    f1_macro_RMS = []
    f1_micro_RMS = []
    accuracy_RMS = []
    f1_macro = []
    f1_micro = []
    accuracy = []
    for i in range(36):
        print("at user {}".format(i))
        trainRMSX, trainX, trainY = getXY(training[i])
        validRMSX, validX, validY = getXY(validation[i])
        testRMSX, testX, testY = getXY(testing[i])
        # combine validatation and training together
        trainRMSX.extend(validRMSX)
        trainX.extend(validX)
        trainY.extend(validY)
        # classes as ints
        trainY = [int(floatclass) for floatclass in trainY]
        testY = [int(floatclass) for floatclass in testY]

        # RMS windows
        # train and test the model
        predictRMSY, _, _ = evaluate_hmm_model(trainRMSX, trainY, testRMSX, testY)
        # evaluate the model
        f1_macro_RMS.append(f1_score(testY, predictRMSY, average='macro'))
        f1_micro_RMS.append(f1_score(testY, predictRMSY, average='micro'))
        accuracy_RMS.append(accuracy_score(testY, predictRMSY))

        # get input in list of list format (RMS inputs are already as such)
        # from 3D
        trainX = pd.concat(trainX)
        testX = pd.concat(testX)
        trainX = trainX.values.tolist()
        testX = testX.values.tolist()
        # trainX.reshape()
        # testX.reshape()
        # print("trainX shape: {}".format(trainX.shape))
        # print("testX shape: {}".format(testX.shape))

        # 200 for each value
        trainY_expand = []
        testY_expand = []
        #for val in trainY:


        # Raw windows
        # train and test the model
        predictY, _, _ = evaluate_hmm_model(trainX, trainY, testX, testY)
        # evaluate the model 
        f1_macro.append(f1_score(testY, predictY, average='macro'))
        f1_micro.append(f1_score(testY, predictY, average='micro'))
        accuracy.append(accuracy_score(testY, predictY))

    print("RMS! Macro-F1: {}, Micro-F1: {}, Accuracy: {}".format(mean(f1_macro_RMS), mean(f1_micro_RMS), mean(accuracy_RMS)))
    # print("RAW! Macro-F1: {}, Micro-F1: {}, Accuracy: {}".format(mean(f1_macro), mean(f1_micro), mean(accuracy)))
    exportCSV(f1_macro_RMS, "hmm_f1_macro_RMS.csv")
    exportCSV(f1_micro_RMS, "hmm_f1_micro_RMS.csv")
    exportCSV(accuracy_RMS, "hmm_accuracy_RMS.csv")
    return


# In[4]:


def evaluate_single_hmm():
    f1_macro_RMS = []
    f1_micro_RMS = []
    accuracy_RMS = []
    f1_macro = []
    f1_micro = []
    accuracy = []
    combineTrainX = []
    combineTrainY = []
    for i in range(36):
        # print("combining data: at user {}".format(i))
        trainRMSX, trainX, trainY = getXY(training[i])
        validRMSX, validX, validY = getXY(validation[i])
        # combine validatation and training together
        trainRMSX.extend(validRMSX)
        trainX.extend(validX)
        trainY.extend(validY)

        combineTrainX.extend(trainRMSX)
        combineTrainY.extend(trainY)
        combineTrainY = [int(floatclass) for floatclass in combineTrainY]

    # train
    # Reorder train data by class
    classes = 6
    x_train_in, _ = group_training_by_class(classes, combineTrainX, combineTrainY)
    # Make models for each class
    hmm_models = train_hmm_models_per_user(classes, x_train_in)

    # test the model for each user
    for i in range(36):
        print("testing: at user {}".format(i))
        testRMSX, testX, testY = getXY(testing[i])
        # classes as ints
        testY = [int(floatclass) for floatclass in testY]

        # RMS windows
        # test the model
        # Classify each sample
        predictRMSY = test_hmm_models_per_user(classes, testRMSX, hmm_models)
        # evaluate the model
        f1_macro_RMS.append(f1_score(testY, predictRMSY, average='macro'))
        f1_micro_RMS.append(f1_score(testY, predictRMSY, average='micro'))
        accuracy_RMS.append(accuracy_score(testY, predictRMSY))
        #print("\t RMS! Macro-F1: {}, Micro-F1: {}, Accuracy: {}".format(mean(f1_macro_RMS), mean(f1_micro_RMS), mean(accuracy_RMS)))

        '''
        # Raw windows
        # train and test the model
        predictY, _, _ = evaluate_hmm_model(trainX, trainY, testX, testY)
        # evaluate the model 
        f1_macro.append(f1_score(testY, predictY, average='macro'))
        f1_micro.append(f1_score(testY, predictY, average='micro'))
        accuracy.append(accuracy_score(testY, predictY))
        '''
    print("RMS! Macro-F1: {}, Micro-F1: {}, Accuracy: {}".format(mean(f1_macro_RMS), mean(f1_micro_RMS), mean(accuracy_RMS)))
    #print("Macro-F1: {}, Micro-F1: {}, Accuracy: {}".format(mean(f1_macro), mean(f1_micro), mean(accuracy)))
    exportCSV(f1_macro_RMS, "hmm_f1_macro_RMS.csv")
    exportCSV(f1_micro_RMS, "hmm_f1_micro_RMS.csv")
    exportCSV(accuracy_RMS, "hmm_accuracy_RMS.csv")
    return





# In[5]:


evaluate_user_hmms()
#evaluate_single_hmm()
