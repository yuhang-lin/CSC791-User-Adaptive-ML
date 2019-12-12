#!/usr/bin/env python
# coding: utf-8

# In[1]:


from preprocessEMG import train_valid_test_split, getXY
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from statistics import mean, stdev
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


from exportCSV import exportCSV


# In[2]:


training, validation, testing = train_valid_test_split()


# In[3]:


def evaluate(clf, clf_name):
    f1_macro = []
    f1_micro = []
    accuracy = []
    for i in range(36):
        trainRMSX, trainX, trainY = getXY(training[i])
        validRMSX, validX, validY = getXY(validation[i])
        # combine validatation and training together
        trainRMSX.extend(validRMSX)
        trainX.extend(validX)
        trainY.extend(validY)
        # train the model
        clf.fit(trainRMSX, trainY)
        # test the model
        testRMSX, testX, testY = getXY(testing[i])
        predictY = clf.predict(testRMSX)
        f1_macro.append(f1_score(testY, predictY, average='macro'))
        f1_micro.append(f1_score(testY, predictY, average='micro'))
        accuracy.append(accuracy_score(testY, predictY))

    print("Macro-F1: {}, Accuracy: {}".format(mean(f1_macro), mean(accuracy)))
    f1_scores_2 = [clf_name, mean(f1_macro), stdev(f1_macro)]
    f1_scores_2.extend(f1_macro)
    accuracies_2 = [clf_name, mean(accuracy), stdev(accuracy)]
    accuracies_2.extend(accuracy)
    exportCSV(f1_scores_2, "full_f1_macro.csv")
    exportCSV(accuracies_2, "full_accuracy.csv")


# In[4]:


clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1) # 1
evaluate(clf, "randomforest")


# In[5]:


clf = ExtraTreesClassifier(n_estimators=200, random_state=0, n_jobs=-1) # 2
evaluate(clf, "extratrees") 


# In[6]:


clf = MLPClassifier(hidden_layer_sizes=(150), max_iter=500, random_state=0) # 3
evaluate(clf, "mlp")


# In[7]:


clf = KNeighborsClassifier(5) # 4
evaluate(clf, "knn")


# In[ ]:




