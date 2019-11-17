#!/usr/bin/env python
# coding: utf-8

# In[11]:


"""
https://pyswarms.readthedocs.io/en/latest/examples/usecases/feature_subset_selection.html
May need features discretized to 2 bins

Issues:
We can't guarantee the feature subset size
- truncate?
- allow for now
Should I be using a continuous PSO optimizer?
Is function a cost or performance

""";


# In[12]:


import numpy as np
import pandas as pd
import MDP_policy
import prepare
import pyswarms as ps
from exportCSV import exportCSV
import pso_pyswarms

import random


# In[13]:


pso_pyswarms.input_filename = "binned_2_reorder.csv"
pso_pyswarms.output_filename = "pso_training_data.csv"
pso_pyswarms.print_filename = "pso_output.csv"
pso_pyswarms.use_ECR = True


# In[ ]:


n_particles_arg = 40
dimensions_arg = 124
iters_arg = 3

val, final_features, final_feature_names = pso_pyswarms.execute_swarm(n_particles_arg, dimensions_arg, iters_arg)
dimensions_arg = final_features.len()
delta = val

while (delta < 2 or dimensions_arg < 3):
    pso_pyswarms.input_filename = "pso_training_data.csv"
    final_val, final_features, final_feature_names = pso_pyswarms.execute_swarm(n_particles_arg, dimensions_arg, iters_arg)
    dimensions_arg = final_features.len()
    delta = val - final_val


# In[ ]:




