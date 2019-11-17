#!/usr/bin/env python
# coding: utf-8

# In[1]:


import MDP_policy


# In[2]:


ECR_value, IS_value = MDP_policy.induce_policy_MDP("best_ecr.csv", True)


# In[3]:


ECR_value, IS_value = MDP_policy.induce_policy_MDP("best_is.csv", True)


# In[ ]:




