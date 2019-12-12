# Self-selected Project

## Data
We used EMG data for gestures Data Set, which is available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip).

## Data prepocessing
We did the following steps for preprocessing:
1. Dropped class 0 and 7 from the dataset
2. Imputed missing values
3. Partitioned data into window sizes of 200ms with stride of 100ms
4. For modeling, we split the data into 50:25:25 for train, validation and testing respectively.


## Requirements for running
sklearn  0.19.1
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
LSTM.py # LSTM
RMS_full_features-Aggregated.py # full feature set, aggregated training
RMS_full_features.py # full feature set, individual training
RMS_limited_features-Aggregated.py # limited feature set, aggregated training
RMS_limited_features.py # limited feature set, individual training
RMSClassificationHMM.py # HMM model using RMS as data
```
