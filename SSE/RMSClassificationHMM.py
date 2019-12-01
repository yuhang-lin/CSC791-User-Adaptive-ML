#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system(' pip install hmmlearn')


# In[3]:

import numpy as np
from preprocessEMG import train_valid_test_split, getXY
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from statistics import mean

from HMMlearn import evaluate_hmm_model
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.neural_network import MLPClassifier


# In[2]:


training, validation, testing = train_valid_test_split()


# In[3]:


def evaluate_hmm():
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

        '''
        # get input in list of list format (RMS inputs are already as such)
        # currently 3D - what to do?
        trainX = np.asarray([X.values.tolist() for X in trainX])
        testX = np.asarray([X.values.tolist() for X in testX])
        # trainX.reshape()
        # testX.reshape()
        print("trainX shape: {}".format(trainX.shape))
        print("testX shape: {}".format(testX.shape))
        '''

        # RMS windows
        # train and test the model
        predictRMSY, _, _ = evaluate_hmm_model(trainRMSX, trainY, testRMSX, testY)
        # evaluate the model
        f1_macro_RMS.append(f1_score(testY, predictRMSY, average='macro'))
        f1_micro_RMS.append(f1_score(testY, predictRMSY, average='micro'))
        accuracy_RMS.append(accuracy_score(testY, predictRMSY))

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
    return

# In[4]:


evaluate_hmm()

