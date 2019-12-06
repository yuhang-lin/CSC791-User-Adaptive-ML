#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

def generate_plots(subject, model_history):
    """
    A method that takes the model history of a trained model and plots its:
    1. Training accuracy
    2. Training loss
    3. Validation accuracy
    4. Validation loss
    """
    acc = model_history.history['acc']
    val_acc = model_history.history['val_accuracy']
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
    #plt.show()
    plt.close()
