import pandas as pd
import numpy as np
from hmmlearn import hmm
import pickle
#import preprocessing
import operator

#import scipy
#from hmmlearn import hmm
#from hmm_classifier import HMM_classifier

'''
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

'''


###
# READ INPUT
###
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


'''
data alt
'''

classes = 6

# sample data
x_train = np.random.randint(1, classes, size=(300, 2)).tolist()
y_train = np.random.randint(1, classes, size=(300)).tolist()
x_test = np.random.randint(1, classes, size=(100, 2)).tolist()
# reorder train & test data by class
# lists (classes) of listoflists (multiple samples)
x_train_in = [[] for _ in range(classes)]
y_train_in = [[] for _ in range(classes)]
x_test_in = [[] for _ in range(classes)]
for idx in range(len(x_train)):
    class_idx = y_train[idx]
    x_train_in[class_idx - 1].append(x_train[idx])
    y_train_in[class_idx - 1].append(y_train[idx])


###
# Make models for each class
###

np.random.seed(42)
'''
# n_components = 6
x_train_lengths = []
for sequence in x_train:
    x_train_lengths.append(len(sequence))
'''
hmm_models = [[] for _ in range(classes)]
for class_idx in range(classes):
    # model does not exist
    if len(x_train_in[class_idx]) != 0:
        hmm_model = hmm.GaussianHMM().fit(x_train_in[class_idx])
        #hmm_model = hmm.MultinomialHMM().fit(x_train_in[class_idx])
        hmm_models[class_idx].append(hmm_model)

###
# Classify each sample
###
y_pred = []
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
    y_pred.append(y_idx + 1)


print("train samples: {}".format(x_train))
print("class labels: {}".format(y_train))
print("test samples: {}".format(x_test))
#y_pred = run_hmm_classifier_example(x_train, y_train, x_test)
#y_pred = run_hmm_model(x_train, y_train, x_test)
print("pred labels samples: {}".format(y_pred))

# evaluate y_pred vs. y_test





'''
# for each user in training data
for user in user_data:
    # load model (from file, if needed)
    # inputmodelname = 'model-{}.pkl'.format(user)
    # with open(inputmodelname, "rb") as file: pickle.load(file)

    # run model


    # evaluate prediction


    # save model (if needed)
    # outputmodelname = 'model-{}.pkl'.format(user)
    # with open(outputmodelname, "wb") as file: pickle.dump(remodel, file)

'''
