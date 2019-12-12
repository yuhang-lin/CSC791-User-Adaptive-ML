#!/usr/bin/env python
# coding: utf-8

# import os
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
# os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

# import numpy as np
# import random as rn
# import tensorflow as tf

# rn.seed(1)
# np.random.seed(1)
# from tensorflow import set_random_seed
# set_random_seed(1)

# from keras import backend as k
# config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
# allow_soft_placement=True, device_count = {'CPU': 1})
# sess = tf.Session(graph=tf.get_default_graph(),config=config)
# k.set_session(sess)

from preprocessEMG import train_valid_test_split, getXY
from evaluateBase import generate_plots

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import numpy as np
from statistics import mean, stdev
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from exportCSV import exportCSV
from keras.models import Sequential
from keras.layers import Bidirectional, ConvLSTM2D, ConvLSTM2D, Dense, Dropout, Flatten, LSTM, TimeDistributed
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

def build_model(n_length, n_features, n_outputs, individual_training=False):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.05)), 
                                  input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    individual_training = True
    if individual_training:
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
    else:
        model.add(LSTM(50, activation='tanh', recurrent_regularizer=l2(0.05), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(50, activation='tanh', recurrent_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(n_outputs, activation='softmax'))
    opt = Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def test_model(subject, model, X_test, y_test, individual_training):
    verbose, batch_size = 0, 32
    window_size, n_features, n_outputs = 200, 8, 6
    individual_training = True
    n_steps = 8
    if individual_training:
        n_steps = 4 
    n_length = window_size // n_steps
    X_test = X_test.reshape(X_test.shape[0], n_steps, n_length, n_features)
    
    predicted_labels = model.predict(X_test, batch_size=batch_size)

    y_pred = [np.argmax(x) for x in predicted_labels]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return [acc, f1]

# fit and evaluate a model
def train_model(subject, X_train, y_train, X_valid, y_valid, epochs=50, individual_training=False):
    """A function that trains an LSTM model and returns the accuracy on the test dataset.

    Parameters
    ----------
    X_train : numpy array
        Training data
    y_train : numpy array
        Training labels
    X_test : numpy array
        Testing data
    y_test : numpy array
        Testing labels

    Returns
    -------
        The accuracy of the trained and evaluated LSTM model.

    """
    verbose, batch_size = 0, 32
    window_size, n_features, n_outputs = 200, 8, 6
    individual_training = True
    n_steps = 8
    if individual_training:
        n_steps = 4  
    n_length = window_size // n_steps
    
    #annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
    es = EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1, restore_best_weights=True)
    #es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.001, patience=15, verbose=1, restore_best_weights=True)
    #mcp_save = ModelCheckpoint('.mdl_wts.hdf5', monitor='val_accuracy', mode='max', save_best_only=True)
    mcp_save = ModelCheckpoint('mdl_wts.hdf5', monitor='val_loss', mode='min')
   
    X_train = X_train.reshape(X_train.shape[0], n_steps, n_length, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], n_steps, n_length, n_features)
    
    
    model = build_model(n_length, n_features, n_outputs, individual_training)
    model.summary()
    # fit network
    hist = model.fit(X_train, y_train, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=[es, mcp_save], validation_data=(X_valid, y_valid))
    return model

training, validation, testing = train_valid_test_split()
results = []
epochs = 160
individual_training = False
if individual_training:
    for i in range(36):
        print("----------------------\n")
        print("Training for user {}\n".format(i+1))
        print("----------------------\n")
        trainRMSX, trainX, trainY = getXY(training[i])
        validRMSX, validX, validY = getXY(validation[i])

        # get the testing data
        testRMSX, testX, testY = getXY(testing[i])

        trainX = np.asarray([X.values for X in trainX])
        trainY = np.asarray(trainY)
        validX = np.asarray([X.values for X in validX])
        validY = np.asarray(validY)
        testX = np.asarray([X.values for X in testX])
        testY = np.asarray(testY)  

        # train and test the model
        model = train_model(i + 1, trainX, trainY, validX, validY, epochs, individual_training)
        results.append(test_model(i + 1, model, testX, testY, individual_training))
else:
    combineTrainX = []
    combineTrainY = []
    combineValidX = []
    combineValidY = []
    # combine all training data together
    for i in range(36):
        trainRMSX, trainX, trainY = getXY(training[i])
        validRMSX, validX, validY = getXY(validation[i])
        combineTrainX.extend(trainX)
        combineTrainY.extend(trainY)
        combineValidX.extend(validX)
        combineValidY.extend(validY)
    combineTrainX = np.asarray([X.values for X in combineTrainX])
    combineTrainY = np.asarray(combineTrainY)
    combineValidX = np.asarray([X.values for X in combineValidX])
    combineValidY = np.asarray(combineValidY)
    model = train_model(i + 1, combineTrainX, combineTrainY, combineValidX, combineValidY, epochs, individual_training)
    for i in range(36):
        testRMSX, testX, testY = getXY(testing[i])
        testX = np.asarray([X.values for X in testX])
        testY = np.asarray(testY)
        results.append(test_model(i + 1, model, testX, testY, individual_training))

accuracies = [val[0] for val in results]
f1_scores = [val[1] for val in results]

experiment_name = "cnnStackedLSTM"

accuracies_2 = [experiment_name, mean(accuracies), stdev(accuracies)]
accuracies_2.extend(accuracies)
f1_scores_2 = [experiment_name, mean(f1_scores), stdev(f1_scores)]
f1_scores_2.extend(f1_scores)

if individual_training:
    exportCSV(accuracies_2, "accuracy_cnnStackedLSTM_indiv.csv")
    exportCSV(f1_scores_2, "f1_cnnStackedLSTM_indiv.csv")
else:
    exportCSV(accuracies_2, "accuracy_cnnStackedLSTM_agg.csv")
    exportCSV(f1_scores_2, "f1_cnnStackedLSTM_agg.csv")

print(accuracies_2)
print(f1_scores_2)

