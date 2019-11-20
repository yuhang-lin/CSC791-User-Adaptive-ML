#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Runs PSO (pso_pyswarms.py) iteratively to reduce feature size to.
Takes reduced feature output of previous run to be further reduced in the next run

Reference:
https://pyswarms.readthedocs.io/en/latest/examples/usecases/feature_subset_selection.html
""";


# In[2]:


import numpy as np
import pandas as pd
import MDP_policy
import prepare
from exportCSV import exportCSV

import pyswarms as ps
import pso_pyswarms
import random


# In[3]:


# MDP file setup
pso_pyswarms.input_filename = "binned_2_reorder.csv"
pso_pyswarms.output_filename = "pso_training_data.csv"
pso_pyswarms.print_filename = "pso_output.csv"
pso_pyswarms.use_ECR = True # True, False


# In[ ]:


# PSO parameters
n_particles_arg = 2     # 40
dimensions_arg = 124
iters_arg = 2           # 10

# Execute
val, final_feature_names, _ = pso_pyswarms.execute_swarm(n_particles_arg, dimensions_arg, iters_arg)
dimensions_arg = final_feature_names.len()
delta = val

while (delta < 2 or dimensions_arg < 3):
    print("current dimensions: {}".format(dimensions_arg))
    pso_pyswarms.input_filename = "pso_training_data.csv"
    final_val, final_feature_names, _ = pso_pyswarms.execute_swarm(n_particles_arg, dimensions_arg, iters_arg)
    dimensions_arg = final_feature_names.len()
    delta = final_val - val 
    print("delta: {}".format(delta))
    val = final_val
print("end of run")

