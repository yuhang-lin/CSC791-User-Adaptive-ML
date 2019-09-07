Before running the code, configuration needs to be done by adapting the code in
mimicConfig_release.R, specifically:

- updating dnsroot to be the folder with the data
- creating the micegp_log folder in the specified `dnsroot`.
- install the following dependencies:
  - MICE
  - GPfit
  - hash
  - doParallel
  - foreach

To train, run the following code directly:

source('mimicMICEGPParamEvalTr_release.R')

or better run as R markdown:
library(rmarkdown)
render('mimicMICEGPParamEvalTr_release.R')

This is a wrapper code calling various subroutines that generate the training data, mask missing values, and performs 3D-MICE imputation, each step is wrapped in its own R source file and should be self-explanatory.

In this wrapper code, nimp specifies how many MICE imputation to perform, ncores specifies how many cores to parallel the multiple imputations. nimp should be a multiply of ncores. You can set ncores to higher or lower values depending on the machine capacity. On a 20 core machine, this code should run in less than a day.
