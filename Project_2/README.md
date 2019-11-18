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
`python3 repeat.py` will call `genetic.py` to run some iterations of genetic algorithm for searching for some features. The log is saved in `bin_NUM_history` folder where NUM is the number of bins.  
Please change the parameters in `repeat.py` to do feature selection using other number of bins and number of features.  

### PSO Algorithm  
`python3 pso_pyswarms_control.py` calls `pso_pyswarms.py` to further run the algorithm with the features that were output in the previous run.  
The parameters are clearly defined in `python3 pso_pyswarms_control.py`.
(However, we abandoned our use of this algorithm due to poor preliminary results).

## Correlation and Heat Map
```cd correlation 
python3 correlation.py``` will generate heat map and do correlation calculation
