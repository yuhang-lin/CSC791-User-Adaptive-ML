#!/usr/bin/env python
# coding: utf-8

import random
import MDP_policy
from prepare import prepare
from exportCSV import exportCSV

num_feature = 6 # number of features
num_generation = 10 # number of generation
num_child = 10 # number of children in one generation
use_ECR = True # selection based on ECR if true
num_mutate = 1 # number of features being mutated at a time


# all possible features indices you can choose
feature_list = [i for i in range(6, 130)]
# get random initial policy
parent = sorted(random.sample(feature_list, num_feature))

def get_value(features, use_ECR):
    filename = "temp.csv"
    prepare(features, "binned_org_order.csv", filename)
    ECR_value, IS_value = MDP_policy.induce_policy_MDP(filename)
    data = [ECR_value, IS_value]
    data.extend(features)
    exportCSV(data, "{}.csv".format(num_feature))
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

parent_value = get_value(parent, use_ECR)
for i in range(num_generation):
    best_value = None
    best_child = None
    for j in range(num_child):
        child_features = mutate(parent, feature_list, num_mutate)
        # prepare CSV based on the selected features
        curr_value = get_value(child_features, use_ECR)
        # get the best ECR or IS among all children
        if best_value is None or best_value < curr_value:
            best_value = curr_value
            best_child = child_features
    # compare it with the parent
    if best_value > parent_value:
        parent_value = best_value
        parent = best_child

print(parent)

