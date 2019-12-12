"""
    Performs  Hidden Markov Model(HMM) model classification
    (using hmmlearn)

    Classifying Hand Gestures from EMG Data
    https://github.com/kunalnarangtheone/CSC591ML/tree/master/SSE

    Group:
    Yuhang Lin          ylin34@ncsu.edu
    Kunal Narang        knarang@ncsu.edu
    Fogo Tunde-Onadele  oatundeo@ncsu.edu
"""

import pandas as pd
import numpy as np
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from hmmlearn import hmm
import operator
#import pickle
#import preprocessing


def group_training_by_class(classes, x_train, y_train):
    """
    Groups training dataset by class.

    Parameters
    ----------
    classes : int
        Number of labels
    x_train : list of lists (outer list -> samples)
        Training data
    y_train : list (list of classes)
        Training labels

    Returns
    -------
    x_train_in : list of listoflists (outer list -> by class -> samples)
        Training data by class
    y_train_in : list of lists (outer list -> classes)
        Training labels by class
    """

    # lists (classes) of listoflists (multiple samples)
    x_train_in = [[] for _ in range(classes)]
    y_train_in = [[] for _ in range(classes)]
    for idx in range(len(x_train)):
        class_idx = int(y_train[idx])
        #x_train_in[class_idx - 1].append(x_train[idx])
        #y_train_in[class_idx - 1].append(y_train[idx])
        x_train_in[class_idx].append(x_train[idx])
        y_train_in[class_idx].append(y_train[idx])
    return x_train_in, y_train_in


def train_hmm_models_per_user(classes, x_train_in):
    """
    Makes class number of hmm models for 1 user.

    Parameters
    ----------
    classes : int
        Number of labels
    x_train_in : list of listoflists (outer list -> by class -> samples)
        Training data by class

    Returns
    -------
    hmm_models : list of hmm models (outer list -> hmm model)
        HMM models
    """
    # GaussianHMM
    # GMMHMM
    num_iterations = 10

    hmm_models = [[] for _ in range(classes)]
    for class_idx in range(classes):
        # model exists
        if len(x_train_in[class_idx]) != 0:
            hmm_model = hmm.GaussianHMM(n_iter=num_iterations).fit(x_train_in[class_idx])
            #hmm_model = hmm.MultinomialHMM().fit(x_train_in[class_idx])
            #hmm_model = hmm.MultinomialHMM(n_components = classes).fit(x_train_in[class_idx])
            hmm_models[class_idx].append(hmm_model)
    '''
    # save model
    outputmodelname = 'model-{}.pkl'.format(user)
    with open(outputmodelname, "wb") as file:
        pickle.dump(remodel, file)
    '''
    return hmm_models


def test_hmm_models_per_user(classes, x_test, hmm_models):
    """
    Outputs prediction of the hmm models for 1 user,
    as well as accuracy and f1-score.

    Parameters
    ----------
    classes : int
        Number of labels
    x_test: list of lists (outer list -> by class -> samples)
        Training data by class
    hmm_models : list of hmm models (outer list -> hmm model)
        HMM models

    Returns
    -------
    y_pred : list of int (outer list -> pred)
        List of prediction labels
    """

    y_pred = []
    '''
    # load model (in another file)
    # inputmodelname = 'model-{}.pkl'.format(user)
    # with open(inputmodelname, "rb") as file: pickle.load(file)
    '''

    for sequence in x_test:
        score_list = [[] for _ in range(classes)]
        # find most likely class model for sequence
        for idx, hmm_model in enumerate(hmm_models):
            if len(hmm_model) != 0:
                score_list[idx].append(hmm_model[0].score([sequence]))
            else:
                score_list[idx].append(float("-infinity"))
        # flatten list
        scores = [item for sublist in score_list for item in sublist]
        y_idx, y_score = max(enumerate(scores), key=operator.itemgetter(1))
        #y_pred.append(y_idx + 1)
        y_pred.append(y_idx)
    return y_pred


def evaluate_hmm_model(x_train, y_train, x_test, y_test, classes=6):
    """
    Groups training dataset by class.

    Parameters
    ----------
    x_train : list of lists (outer list -> samples)
        Training data
    y_train : list (list of classes)
        Training labels
    x_test: list of lists (outer list -> by class -> samples)
        Training data by class
    y_test : list (list of classes)
        Testing labels
    classes : int
        Number of labels

    Returns
    -------
    y_pred : list of int (outer list -> pred)
        List of prediction labels
    acc : float
        Accuracy per user
    f1 : float
        F1 score per user
    """
    # Reorder train data by class
    x_train_in, _ = group_training_by_class(classes, x_train, y_train)
    # Make models for each class
    hmm_models = train_hmm_models_per_user(classes, x_train_in)
    # Classify each sample
    y_pred = test_hmm_models_per_user(classes, x_test, hmm_models)

    # Evaluate y_pred vs. y_test
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return y_pred, acc, f1

def run_example():
    # READ INPUT

    #train_inputX = preprocessing.read_user_data()
    '''
    train = pd.read_csv("sample_data.txt", sep = "\t")
    train_row, train_col = train.shape
    x_train = train.iloc[:train_row - 2]
    y_train = train.iloc[train_row - 2]
    #len(train_inputX)
    test = pd.read_csv("sample_data2.txt", sep = "\t")
    test_row, test_col = test.shape
    x_test = test.iloc[:test_row - 2]
    test_len = len(test_inputX)
    y_test = test.iloc[test_row - 2]
    '''
    # sample data
    np.random.seed(42)

    classes = 6
    x_train = np.random.randint(1, classes, size=(300, 2)).tolist()
    y_train = np.random.randint(1, classes, size=(300)).tolist()
    x_test = np.random.randint(1, classes, size=(100, 2)).tolist()
    y_test = np.random.randint(1, classes, size=(100)).tolist()

    # Run HMM
    y_pred, acc, f1 = evaluate_hmm_model(x_train, y_train, x_test, y_test)

    print("train samples: {}".format(x_train))
    print("class labels: {}".format(y_train))
    print("test samples: {}".format(x_test))
    print("pred labels samples: {}".format(y_pred))
    print("acc: {}, f1_score: {}".format(acc, f1))
    return

'''
# HMM classifier wrapper (issues)

#import scipy
#from hmmlearn import hmm
#from hmm_classifier import HMM_classifier

x_train = np.random.randint(1, classes, size=(300, 2))
y_train = np.random.randint(1, classes, size=(300))
x_test = np.random.randint(1, classes, size=(100, 2))
y_test = np.random.randint(1, classes, size=(300))

def run_hmm_classifier_example(X_train, y_train, X_test):
    model = HMM_classifier(hmm.MultinomialHMM())
    model.fit(X_train,y_train)

    # Predict probability per label
    #pred = model.predict_proba(np.random.randint(0, 10, size=(10, 2)))

    # Get label with the most high probability
    y_pred = model.predict(X_test)
    print("pred labels: {}".format(y_pred))

    # evaluate y_pred vs. y_test

    return y_pred
    
y_pred = run_hmm_classifier_example(x_train, y_train, x_test)

'''
