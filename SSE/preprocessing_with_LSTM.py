# -*- coding: utf-8 -*-

from preprocessEMG import train_valid_test_split, getXY

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from statistics import mean



training, validation, testing = train_valid_test_split()



# fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test):
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
    verbose, epochs, batch_size = 0, 200, 32
    n_timesteps, n_features, n_outputs = 200, 8, 6
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(n_timesteps,n_features)))    
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    model.summary()
    # fit network
    
    with tf.device('/device:GPU:0'):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # evaluate model
    # _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

    predicted_labels = model.predict(X_test, batch_size=batch_size)

    y_pred = [np.argmax(x) for x in predicted_labels]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="micro")

    return [acc, f1]


results = []
for i in range(36):
    print("----------------------\n")
    print("Training for user {}\n".format(i+1))
    print("----------------------\n")
    trainRMSX, trainX, trainY = getXY(training[i])
    validRMSX, validX, validY = getXY(validation[i])
    # combine validatation and training together
    trainX.extend(validX)
    trainY.extend(validY)

    # get the testing data
    testRMSX, testX, testY = getXY(testing[i])
    
    trainX = np.asarray([X.values for X in trainX])
    trainY = np.asarray(trainY)
    testX = np.asarray([X.values for X in testX])
    testY = np.asarray(testY)

    # train and test the model
    results.append(evaluate_model(trainX, trainY, testX, testY))



accuracies = [val[0] for val in results]
f1_scores = [val[1] for val in results]

f = open("results.txt", "w+")

f.write("Accuracies\n")
f.write("----------------------\n")
f.write("{}\n".format(accuracies))
f.write("\n")

f.write("F1 scores\n")
f.write("----------------------\n")
f.write("{}\n".format(f1_scores))
f.write("\n")

f.write("----------------------\n")
f.write("----------------------\n")
f.write("\n")

f.write("Average accuracy over all users\n")
f.write("----------------------\n")
f.write("{}\n".format(mean(accuracies)))
f.write("\n")

f.write("Average F1 scores over all users\n")
f.write("----------------------\n")
f.write("{}\n".format(mean(f1_scores)))
f.write("\n")

f.close()


