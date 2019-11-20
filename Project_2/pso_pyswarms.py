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

""";


# In[2]:


'''
%%capture
!pip3 install pyswarms
'''


# In[3]:


import numpy as np
import pandas as pd
import MDP_policy
import prepare
import pyswarms as ps
from exportCSV import exportCSV

import random
# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


RANDOM_SEED = 56
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[5]:


input_filename = "binned_2_reorder.csv"
output_filename = "pso_training_data.csv"
use_ECR = True
print_filename = "pso_output.csv"


# In[6]:


# optimize MDP_policy.induce_policy_MDP()
# make negative if function is a cost
def get_value_func_per_particle(input_feature_masks):
    # input_feature_masks type: np ndarray
    
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


# In[7]:


# outer function
def get_value_func_outer(particles):
    # particles shape: particles x dimensions    
    
    n_particles = particles.shape[0]
    j = [get_value_func_per_particle(particles[i]) for i in range(n_particles)]
    return np.array(j)


# In[8]:


def execute_swarm(n_particles_arg, dimensions_arg, iters_arg):
    # Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.7, 'k': 35, 'p':2}
    #options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO
    # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=35, dimensions=dimensions_arg, options=options)
    #optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options)

    # Perform optimization
    output_cost, output_pos = optimizer.optimize(get_value_func_outer, iters=iters_arg)

    # Output chosen features
    final_val = -output_cost
    final_feature_pos = np.squeeze(np.where(output_pos)).tolist()
    #print("pos {}".format(final_feature_pos))
    checked_feature_names = pd.read_csv(input_filename).columns.tolist()

    final_feature_names = []
    for i in range(len(final_feature_pos)):
        feature_index = final_feature_pos[i]
        #print("index {}".format(feature_index))
        final_feature_names.append(checked_feature_names[feature_index])
        #print("collection {}".format(final_feature_names))

    exportCSV([final_val, final_feature_names], fileName=print_filename)
    print("Value: {}".format(final_val))
    #print("Feature pos: {}".format(final_feature_pos))
    print("Features names selected:\n {}".format(final_feature_names))
    #print("Feature names considered:\n {}".format(checked_feature_names))
    
    return final_val, final_feature_names, checked_feature_names