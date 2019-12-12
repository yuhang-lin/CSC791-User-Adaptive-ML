# Self-selected Project

## Data
We used EMG data for gestures Data Set, which is available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip).

## Data prepocessing

## Requirements for running
sklearn  
keras  
tensorflow > 1.13.1  
hmmlearn  
python > 3.0  

## How to run main models
```
python3 cnn.py # CNN 
python3 cnnLSTM.py # CNN + LSTM
python3 cnnBiLSTM.py # CNN + Bidirectional LSTM
python3 cnnStackedLSTM.py # CNN + Stacked LSTM
```
By default, the main models use aggreagated training, which performs better than invidual training according to our experiments.
However, you can also train individual models if you set individual_training=True in the code.

## Baseline models
```
LSTM
KNN
Random Forest
MLP
Extremely Randomized Trees
```
