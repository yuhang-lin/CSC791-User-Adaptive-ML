# Assigned Project 2

## Final Policy Evaluation
### ECR
`python3 MDP_policy.py -input Training_data.csv` will run the MDP program and get the best ECR value along with its policy. 
### IS
`python3 MDP_policy.py -input Training_data_is.csv` will run the MDP program and get the best IS value along with its policy. 

## All Data Used
`binned_{2,3,4,5}_reorder.csv` 

## Feature Discretization

## Feature Selection 
### Genetic Algorithm
`python3 repeat.py` will call `genetic.py` to run some iterations of genetic algorithm to select features. Each unique new feature list saved to a log file in `bin_NUM_history` folder where NUM is the number of bins. Please change the parameters in `repeat.py` to do feature selection using other number of bins and number of features.

## Correlation and Heat Map
```cd correlation 
python3 correlation.py``` will generate heat map and do correlation calculation
