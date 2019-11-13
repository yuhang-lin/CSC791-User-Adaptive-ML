#!/usr/bin/env python
# coding: utf-8

import random
import MDP_policy
from prepare import prepare
from exportCSV import exportCSV
import os

def get_history(num_feature):
    """
    Reading the csv file and save the order history record of the features
    :param num_feature:
    :return: map of history (key: features, values: list of ECR,IS values)
    """
    hist = dict()
    hist_file = "{}.csv".format(num_feature)
    if not os.path.isfile(hist_file):
        return hist
    with open(hist_file) as fin:
        for line in fin:
            parts = line.strip().split(",")
            hist[",".join(parts[2:])] = [float(parts[0]), float(parts[1])]
    return hist

def get_value(hist, features, use_ECR):
    ECR_value = None
    IS_value = None
    key = ",".join([str(i) for i in features])
    if key in hist:
        ECR_value = hist[key][0]
        IS_value = hist[key][1]
    else:
        filename = "temp.csv"
        prepare(features, "binned_2_reorder.csv", filename)
        ECR_value, IS_value = MDP_policy.induce_policy_MDP(filename)
        data = [ECR_value, IS_value]
        data.extend(features)
        exportCSV(data, "{}.csv".format(len(features)))
    curr_value = ECR_value
    if not use_ECR:
        curr_value = IS_value
    return curr_value

def mutate(parent, feature_list, num_mutate):
    num_mutate = min(num_mutate, len(parent))
    choice_list = [e for e in feature_list if e not in parent]
    child_features = parent[:]
    new_features = random.sample(choice_list, num_mutate)
    # select random indices to be mutated 
    replace_indices = random.sample([i for i in range(len(parent))], num_mutate)
    for i in range(num_mutate):
        child_features[i] = new_features[i]
    return sorted(child_features)


def main(num_feature=8, num_generation=10, use_ECR=True, num_child=15, num_mutate=2):
    """
    function which should be called to invoke the genetic algorithm
    :param num_feature: Number of features
    :param num_generation: Number of generation
    :param use_ECR: Selection based on ECR if true
    :param num_child: Number of children in one generation
    :param num_mutate: Number of features being mutated at a time
    :return:
    """

    num_feature = max(min(num_feature, 8), 1) # set a max value of 8 and min value of 1
    # all possible features indices you can choose
    feature_list = [i for i in range(6, 130)]

    # get random initial policy
    parent = sorted(random.sample(feature_list, num_feature))

    hist = get_history(num_feature)
    parent_value = get_value(hist, parent, use_ECR)
    for i in range(num_generation):
        print("Generation {}, parent value: {}".format(i, parent_value))
        best_value = None
        best_child = None
        for j in range(num_child):
            child_features = mutate(parent, feature_list, num_mutate)
            # prepare CSV based on the selected features
            curr_value = get_value(hist, child_features, use_ECR)
            # get the best ECR or IS among all children
            if best_value is None or best_value < curr_value:
                best_value = curr_value
                best_child = child_features
        # compare it with the parent
        if best_value > parent_value:
            parent_value = best_value
            parent = best_child
    print(parent)
    if use_ECR:
        print("Best ECR value: {}".format(parent_value))
    else:
        print("Best IS value: {}".format(parent_value))

