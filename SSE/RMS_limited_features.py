#!/usr/bin/env python
# coding: utf-8

# In[3]:


from preprocessEMG import train_valid_test_split, getXY
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from statistics import mean, stdev
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


from exportCSV import exportCSV


# In[4]:


training, validation, testing = train_valid_test_split()


# In[5]:


def selectLimitFeatures(RMS):
    indices = [1, 3, 5, 7]
    limitedRMS = []
    for item in RMS:
        limitedRMS.append([item[index] for index in indices])
    return limitedRMS

def evaluate(clf, clf_name):
    f1_macro = []
    f1_micro = []
    accuracy = []
    for i in range(36):
        trainRMSX, trainX, trainY = getXY(training[i])
        validRMSX, validX, validY = getXY(validation[i])
        testRMSX, testX, testY = getXY(testing[i])
        # combine validatation and training together
        trainRMSX.extend(validRMSX)
        trainX.extend(validX)
        trainY.extend(validY)
        
        # use limited features
        trainRMSX = selectLimitFeatures(trainRMSX)
        testRMSX = selectLimitFeatures(testRMSX)
        
        # train the model
        clf.fit(trainRMSX, trainY)
        # test the model
        
        predictY = clf.predict(testRMSX)
        f1_macro.append(f1_score(testY, predictY, average='macro'))
        accuracy.append(accuracy_score(testY, predictY))

    print("Macro-F1: {}, Accuracy: {}".format(mean(f1_macro), mean(accuracy)))
    f1_scores_2 = [clf_name, mean(f1_macro), stdev(f1_macro)]
    f1_scores_2.extend(f1_macro)
    accuracies_2 = [clf_name, mean(accuracy), stdev(accuracy)]
    accuracies_2.extend(accuracy)
    exportCSV(f1_scores_2, "limited_f1_macro.csv")
    exportCSV(accuracies_2, "limited_accuracy.csv")


# In[6]:


clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1) # 1
evaluate(clf, "randomforest")


# In[7]:


clf = ExtraTreesClassifier(n_estimators=200, random_state=0, n_jobs=-1) # 2
evaluate(clf, "extratrees") 


# In[8]:


clf = MLPClassifier(hidden_layer_sizes=(150), max_iter=500, random_state=0) # 3
evaluate(clf, "mlp")


# In[9]:


clf = KNeighborsClassifier(5) # 4
evaluate(clf, "knn")


# In[ ]:




