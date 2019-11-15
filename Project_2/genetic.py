#!/usr/bin/env python
# coding: utf-8

import random
import MDP_policy
from prepare import prepare
from exportCSV import exportCSV
import os
import gc
from timeit import default_timer

class Genetic():
    def __init__(self):
        # all possible features indices you can choose
        self.feature_list = [i for i in range(6, 130)]
        self.hist = dict()
    
    def count_time(self, last_time, message):
        diff = default_timer() - last_time
        print("{}: {}".format(message, diff))
        last_time += diff
        return last_time

    def get_history(self, num_feature):
        """
        Reading the csv file and save the order history record of the features
        :param num_feature:
        :return: map of history (key: features, values: list of ECR,IS values)
        """
        self.hist = dict()
        hist_file = "{}.csv".format(num_feature)
        if not os.path.isfile(hist_file):
            return
        with open(hist_file) as fin:
            for line in fin:
                parts = line.strip().split(",")
                self.hist[",".join(parts[2:])] = [float(parts[0]), float(parts[1])]

    def get_value(self, features, use_ECR):
        ECR_value = None
        IS_value = None
        key = ",".join([str(i) for i in features])
        if key in self.hist:
            ECR_value = self.hist[key][0]
            IS_value = self.hist[key][1]
        else:
            filename = "temp.csv"
            prepare(features, "binned_3_reorder.csv", filename)
            ECR_value, IS_value = MDP_policy.induce_policy_MDP(filename)
            data = [ECR_value, IS_value]
            data.extend(features)
            exportCSV(data, "{}.csv".format(len(features)))
        curr_value = ECR_value
        if not use_ECR:
            curr_value = IS_value
        return curr_value

    def mutate(self, single, max_mutate):
        # make sure the number of mutation is no more than the length of the features
        num_mutate = random.randint(1, min(max_mutate + 1, len(single)))
        choice_list = [e for e in self.feature_list if e not in single]
        if num_mutate > 0:
            new_features = random.sample(choice_list, num_mutate)
            # select random indices to be mutated 
            replace_indices = random.sample([i for i in range(len(single))], num_mutate)
            for i in range(num_mutate):
                single[i] = new_features[i]
        return sorted(single)

    def mutate_all(self, population, num_parent, max_mutate):
        for i in range(num_parent, len(population)):
            population[i] = self.mutate(population[i], max_mutate)

    def selection(self, population, num_parent, use_ECR):
        values = [self.get_value(population[i], use_ECR) for i in range(len(population))]
        population, values = (list(x) for x in zip(*sorted(zip(population, values), key=lambda pair: pair[1], reverse=True)))
        return population, values

    def crossover(self, population, num_parent):
        for i in range(num_parent, len(population)):
            parent1 = population[i % num_parent]
            parent2 = population[(i + 1) % num_parent]
            cross_point = random.randint(1, len(population[0]) - 1)
            population[i] = parent1[:cross_point]
            population[i].extend(parent2[cross_point:])
            population[i].sort()


    def main(self, num_feature=8, num_parent=3, num_generation=10, use_ECR=True, population_size=12, max_mutate=3):
        """
        function which should be called to invoke the genetic algorithm
        :param num_feature: Number of features
        :param num_parent: Number of parents to keep
        :param num_generation: Number of generation
        :param use_ECR: Selection based on ECR if true
        :param population_size: Size of one generation
        :param max_mutate: Maximum number of features being mutated at a time
        :return:
        """

        num_feature = max(min(num_feature, 8), 1) # set a max value of 8 and min value of 1
        
        self.get_history(num_feature)
        # get random initial policy
        population = [sorted(random.sample(self.feature_list, num_feature)) for i in range(population_size)]
        population, last_values = self.selection(population, num_parent, use_ECR)
        last_time = default_timer()
        gc_frequency = 5 # frequency of calling garbage collection
        for i in range(num_generation):
            if i % gc_frequency == gc_frequency - 1:
                gc.collect()
            last_time = self.count_time(last_time, "")
            print("Generation {}, best value: {}".format(i, last_values[0]))
            self.crossover(population, num_parent)
            self.mutate_all(population, num_parent, max_mutate)
            population, last_values = self.selection(population, num_parent, use_ECR)

        print(population[0])
        if use_ECR:
            print("Best ECR value: {}".format(last_values[0]))
        else:
            print("Best IS value: {}".format(last_values[0]))
