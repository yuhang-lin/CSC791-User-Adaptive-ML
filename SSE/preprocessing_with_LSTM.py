# -*- coding: utf-8 -*-

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


training, validation, testing = train_valid_test_split()

def generate_plots(subject, model_history):
    """
    A method that takes the model history of a trained model and plots its:
    1. Training accuracy
    2. Training loss
    3. Validation accuracy
    4. Validation loss
    """
    acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = len(acc)
    
    plt.figure(1 + int(subject))
    plt.suptitle('Accuracy learning curve', fontsize=20)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.plot(acc, label='training accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.xticks(np.arange(0, epochs, epochs/10))
    plt.legend(loc="lower right")
    os.makedirs("fig_lstm", exist_ok=True)
    plt.savefig("fig_lstm/accuracy{}.png".format(subject), dpi=300)
    
    plt.figure(200 + int(subject))
    plt.suptitle('Loss learning curve', fontsize=20)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.plot(loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.xticks(np.arange(0, epochs, epochs/10))
    plt.legend(loc="upper right")
    plt.savefig("fig_lstm/loss{}.png".format(subject), dpi=300)
    plt.show()



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
    n_timesteps, n_features, n_outputs = 200, 8, 6
    
    #annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
    #es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=15, verbose=1, restore_best_weights=True)
    #es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.001, patience=15, verbose=1, restore_best_weights=True)
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', monitor='val_acc', mode='max', save_best_only=True)
    #mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
   
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(n_timesteps,n_features)))    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])
    model.summary()
    # fit network
    
    with tf.device('/device:GPU:0'):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[mcp_save], 
                            validation_data=(X_valid, y_valid))
        
    # evaluate model
    # _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    model.load_weights(filepath = '.mdl_wts.hdf5')
    predicted_labels = model.predict(X_test, batch_size=batch_size)

    y_pred = [np.argmax(x) for x in predicted_labels]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="micro")
    
    generate_plots(subject, history)

    return [acc, f1]


results = []
epochs = 100
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

f1_scores.insert(0, "lstm")
accuracies.insert(0, "lstm")
exportCSV(f1_scores, "f1_micro_lstm.csv")
exportCSV(accuracies, "accuracy_lstm.csv")
print(f1_scores)
print(accuracies)