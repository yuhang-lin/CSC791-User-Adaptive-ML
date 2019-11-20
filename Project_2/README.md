# Assigned Project 2

## Final Policy Evaluation
### ECR
`python3 MDP_policy.py -input Training_data.csv` will run the MDP program and get the best ECR value along with its policy. 
### IS
`python3 MDP_policy.py -input Training_data_is.csv` will run the MDP program and get the best IS value along with its policy. 

## Necessary packages
```
numpy
pandas
pymdptoolbox
seaborn
matplotlib
pyswarms
sklearn
```

## All Data Used
`binned_{2,3,4,5}_reorder.csv` 

## All Logs
All the CSV files in `bin_{2,3,4,5}_history` 

## Feature Discretization
` python3 discretize.py` will discretize the columns. If you got the error of `ImportError: cannot import name 'KBinsDiscretizer'`, please upgrade your `sklearn` to the latest version (at least >= 0.20).

## Feature Selection 
### Genetic Algorithm
`python3 repeat.py` will call `genetic.py` to run some iterations of genetic algorithm to select up to 8 features. The log is saved in `bin_NUM_history` folder where NUM is the number of bins from 2 to 5. When it runs, a CSV file named `temp.csv` will be created as input to `MDP_policy.py` to evaluate the selected features.  
Please change the parameters in `repeat.py` to do feature selection using other number of bins and number of features.  

### PSO Algorithm  
`python3 pso_pyswarms_control.py` calls `pso_pyswarms.py` to further run the algorithm with the features that were output in the previous run.  
The parameters are clearly defined in `python3 pso_pyswarms_control.py`.
(However, we abandoned our use of this algorithm due to poor preliminary results).

## Correlation and Heat Map
```
cd correlation 
python3 correlation.py
```   
will generate heat map and do correlation calculation
