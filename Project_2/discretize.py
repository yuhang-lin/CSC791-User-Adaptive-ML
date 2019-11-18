#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("MDP_Original_data.csv")


# In[3]:


d_columns = [
'Interaction',
'hintCount',
'TotalTime',
'TotalPSTime',
'TotalWETime',
'avgstepTime',
'avgstepTimePS',
'stepTimeDeviation',
'englishSymbolicSwitchCount',
'Level',
'UseCount',
'PrepCount',
'MorphCount',
'OptionalCount',
'InterfaceErrorCount',
'RightApp',
'WrongApp',
'WrongSemanticsApp',
'WrongSyntaxApp',
'PrightAppRatio',
'RrightAppRatio',
'F1Score',
'FDActionCount',
'BDActionCount',
'DirectProofActionCount',
'InDirectProofActionCount',
'actionCount',
'UseWindowInfo',
'NonPSelements',
'AppCount',
'AppRatio',
'hintRatio',
'BlankRatio',
'HoverHintCount',
'SystemInfoHintCount',
'NextStepClickCountWE',
'PreviousStepClickCountWE',
'deletedApp',
'ruleScoreMP',
'ruleScoreDS',
'ruleScoreSIMP',
'ruleScoreMT',
'ruleScoreADD',
'ruleScoreCONJ',
'ruleScoreHS',
'ruleScoreCD',
'ruleScoreDN',
'ruleScoreDEM',
'ruleScoreIMPL',
'ruleScoreCONTRA',
'ruleScoreEQUIV',
'ruleScoreCOM',
'ruleScoreASSOC',
'ruleScoreDIST',
'ruleScoreABS',
'ruleScoreEXP',
'ruleScoreTAUT',
'cumul_Interaction',
'cumul_hintCount',
'cumul_TotalTime',
'cumul_TotalPSTime',
'cumul_TotalWETime',
'cumul_avgstepTime',
'cumul_avgstepTimeWE',
'cumul_avgstepTimePS',
'cumul_symbolicRepresentationCount',
'cumul_englishSymbolicSwitchCount',
'cumul_UseCount',
'cumul_PrepCount',
'cumul_MorphCount',
'cumul_OptionalCount',
'cumul_InterfaceErrorCount',
'cumul_RightApp',
'cumul_WrongApp',
'cumul_WrongSemanticsApp',
'cumul_WrongSyntaxApp',
'cumul_PrightAppRatio',
'cumul_RrightAppRatio',
'cumul_F1Score',
'cumul_FDActionCount',
'cumul_BDActionCount',
'cumul_DirectProofActionCount',
'cumul_InDirectProofActionCount',
'cumul_actionCount',
'cumul_UseWindowInfo',
'cumul_NonPSelements',
'cumul_AppCount',
'cumul_AppRatio',
'cumul_hintRatio',
'cumul_BlankRatio',
'cumul_HoverHintCount',
'cumul_SystemInfoHintCount',
'cumul_NextStepClickCountWE',
'cumul_PreviousStepClickCountWE',
'cumul_deletedApp',
'CurrPro_NumProbRule',
'CurrPro_avgProbTime',
'CurrPro_avgProbTimePS',
'CurrPro_avgProbTimeDeviationPS',
'CurrPro_avgProbTimeWE',
'CurrPro_avgProbTimeDeviationWE',
'CurrPro_medianProbTime']


discretize_columns = df[d_columns]
discretize_columns


# In[4]:


df.drop(d_columns, inplace=True, axis=1)

# In[5]:


N_BINS = 2

from sklearn.preprocessing import KBinsDiscretizer
bins = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')
binned_columns = bins.fit_transform(discretize_columns)
binned_columns


# In[6]:





# In[7]:


binned_columns = pd.DataFrame(binned_columns)
binned_columns.columns = d_columns
binned_columns = binned_columns.astype('int32')


# In[8]:



# In[9]:



# In[10]:


df = pd.concat([df, binned_columns], axis=1)


# In[11]:


df.to_csv('binned_data_{}.csv'.format(N_BINS), index=False)






