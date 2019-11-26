import tensorflow as tf

def encode_data(y_train, y_test):
    """Encode the labels to their categorical form for use in the model

    Parameters
    ----------
    y_train : numpy array
        Training labels (0-8)
    y_test : numpy array
        Testing labels (0-8)

    Returns
    -------
        A one hot categorical encoding of train and test labels

    """
    train_labels = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='int32')
    test_labels = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='int32')
    return train_labels, test_labels



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
    verbose, epochs, batch_size = 0, 15, 256
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return accuracy
