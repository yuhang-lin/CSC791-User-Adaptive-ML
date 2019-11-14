#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
https://pyswarms.readthedocs.io/en/latest/examples/usecases/feature_subset_selection.html
May need features discretized to 2 bins

Issues:
We can't guarantee the feature subset size
- truncate?
- allow for now
Should I be using a continuous PSO optimizer?
Is function a cost or performance

"""


# In[2]:


#get_ipython().system('pip3 install pyswarms;')


# In[3]:


import numpy as np
import pandas as pd
import MDP_policy
import prepare
import random

import pyswarms as ps


# In[4]:


'''
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
'''


# In[ ]:


# optimize MDP_policy.induce_policy_MDP()
# make negative if function is a cost
def get_value_func_per_particle(input_feature_masks, use_ECR=True):
    # input_feature_masks type: np ndarray
    input_filename = "MDP_Original_data.csv"
    output_filename = "pso_training_data.csv"
    
    # set up features from mask
    #print(input_feature_masks)
    input_feature_indices = np.where(input_feature_masks)
    input_features = np.squeeze(np.add(input_feature_indices, 6))
    #print(input_features)
    
    # TODO: deal with max
    prepare.prepare_allow_over_max(input_features, input_filename, output_filename)
    #prepare(input_features, "MDP_Original_data.csv", output_filename)
    
    # calculate cost
    ECR_val, IS_val = MDP_policy.induce_policy_MDP(output_filename)
    result_val = ECR_val
    if not use_ECR:
        result_val = IS_val
    return -1*result_val
    '''
    # Compute other objective function
    # to incorporate tradeoff btwn performance and num of features 
    alpha = 0.88
    total_features = 124
    input_feature_indices
    j = (alpha * (300 - result_val)
        + (1.0 - alpha) * (1 - (input_feature_indices.shape[1] / total_features)))
    return j
    '''


# In[ ]:


# outer function
def get_value_func_outer(particles):
    # particles shape: particles x dimensions    
    
    n_particles = particles.shape[0]
    j = [get_value_func_per_particle(particles[i]) for i in range(n_particles)]
    return np.array(j)


# In[7]:


# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 10, 'p':2}

# Call instance of PSO
dimensions = 124 # dimensions should be the number of features
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
output_cost, output_pos = optimizer.optimize(get_value_func_outer, iters=5)

# Output chosen features 
print("Value: %.4f"%(output_cost))
print("Feature pos: " + output_pos)


# In[ ]:




