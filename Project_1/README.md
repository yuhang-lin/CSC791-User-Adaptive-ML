# CSC591ML

Before you start running this project, please unzip Baseline.zip, train_data.zip and test_data.zip under the directory of the code. 

To reproduce the results:  
1. Impute the data using random forest. (It will take about 4 hours to run. The output will be in test_output/iterative_imputer_extratrees_iter40)
```python3 random_forest_test.py```

2. Impute the data using 3D-mice. (It will take about 14 hours to run.)  
Run ```./prepare.sh``` to reindex all the test files to start from 1 then move them to the 'train_with_missing' folder.
```cd```to the directory of 3D-mice (could be Basline/code). Change the content of dnsroot in 'mimicConfig_release.R', then type ```R``` to enter the R environment. 
```
set.seed(100)
source('mimicMICEGPParamEvalTr_release.R')
``` 
to start running 3D-mice. 

After the program stops, press ```q()``` to quit the R environment. 
```cd```to the directory of 3D-mice (could be Basline/code) and enter ```R``` to get into the R environment again to export the results to CSV files.
```
load("tr_res_iter2.RData")
for (i in seq(1, 8267, by=1)){ write.csv(t(res$t.imp[[i]]), sprintf("%d.csv", i), row.names = FALSE)}
```
You will get a list of CSV files with names from 1 to 8267. Then run ```./finish.sh``` to reindex the last 2267 files and move them to 
'test_output/3d-mice-test' folder. 

3. Mix the output of them.
Run ```python3 mix_output.py``` to get a mixture of the output from the above two models. The output will be stored as 'test_output/mice_extratrees_40'.
