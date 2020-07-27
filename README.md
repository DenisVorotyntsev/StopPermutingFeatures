# Stop permuting features 

This repository contains code for experiment described in my blog post on permutation importance - 
[Stop permuting features](google.com).

Structure of the project: 

`src` - all experiment's code 

`notebooks`
* `notebooks/0-experiment-illustration.ipynb` - illustration fo the single experiment 
* `notebooks/1-exploring-results.ipynb` - notebooks analysing results of experiment without relearning
* `notebooks/1.1-exploring-results-relearn.ipynb` - notebooks analysing results of experiment with relearning
* `notebooks/2-extrapolation-illustration.ipynb` - notebook illustrating extrapolation problem of permutation importance

`data`
* `data/experiment_results_no_relearn.csv` - combined data from experiment without relearning (1200 runs)
* `data/experiment_results_relearn.csv` - combined data from experiment with relearning (120 runs)
