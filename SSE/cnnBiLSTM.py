#!/usr/bin/env python
# coding: utf-8

from nn-base import generate_plots
from preprocessEMG import train_valid_test_split, getXY

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from statistics import mean
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from exportCSV import exportCSV
from keras.layers import ConvLSTM2D
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# fit and evaluate a model
def evaluate_model(subject, X_train, y_train, X_valid, y_valid, X_test, y_test, epochs=50):
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
    n_steps = 4
    n_length = window_size // n_steps
    
    #annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
    #es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=15, verbose=1, restore_best_weights=True)
    #es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.001, patience=15, verbose=1, restore_best_weights=True)
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', monitor='val_accuracy', mode='max', save_best_only=True)
    #mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
   
    X_train = X_train.reshape(X_train.shape[0], n_steps, n_length, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], n_steps, n_length, n_features)
    X_test = X_test.reshape(X_test.shape[0], n_steps, n_length, n_features)
    
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # fit network
    
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[mcp_save], validation_data=(X_valid, y_valid))
    
#     print("printing keys in the hist")
#     for key in hist.history:
#         print(key)
    
    # evaluate model
    # _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    model.load_weights('.mdl_wts.hdf5')
    predicted_labels = model.predict(X_test, batch_size=batch_size)

    y_pred = [np.argmax(x) for x in predicted_labels]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    #generate_plots(subject, hist)

    return [acc, f1]

training, validation, testing = train_valid_test_split()
results = []
epochs = 200
for i in range(36):
    print("----------------------\n")
    print("Training for user {}\n".format(i+1))
    print("----------------------\n")
    trainRMSX, trainX, trainY = getXY(training[i])
    validRMSX, validX, validY = getXY(validation[i])
    # combine validatation and training together
    #trainX.extend(validX)
    #trainY.extend(validY)

    # get the testing data
    testRMSX, testX, testY = getXY(testing[i])
    
    trainX = np.asarray([X.values for X in trainX])
    trainY = np.asarray(trainY)
    validX = np.asarray([X.values for X in validX])
    validY = np.asarray(validY)
    testX = np.asarray([X.values for X in testX])
    testY = np.asarray(testY)
    

    # train and test the model
    results.append(evaluate_model(i + 1, trainX, trainY, validX, validY, testX, testY, epochs))
    

accuracies = [val[0] for val in results]
f1_scores = [val[1] for val in results]

f1_scores.insert(0, "cnnlstm")
accuracies.insert(0, "cnnlstm")


exportCSV(f1_scores, "f1_cnnlstm.csv")
exportCSV(accuracies, "accuracy_cnnlstm.csv")
print(f1_scores)
print(accuracies)


