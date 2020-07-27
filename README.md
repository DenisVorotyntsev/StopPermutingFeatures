# Stop Permuting Features 

This repository contains code for the experiment described in my blog post on permutation importance - 
[Stop permuting features](https://towardsdatascience.com/stop-permuting-features-c1412e31b63f).

### Structure of the project 

`src` - all the experiment's code 

`notebooks`
* `notebooks/0-experiment-illustration.ipynb` - illustration fo the single experiment 
* `notebooks/1-exploring-results.ipynb` - notebooks analysing results of experiment without relearning
* `notebooks/1.1-exploring-results-relearn.ipynb` - notebooks analysing results of experiment with relearning
* `notebooks/2-extrapolation-illustration.ipynb` - notebook illustrating extrapolation problem of permutation importance

`data`
* `data/experiment_results_no_relearn.csv` - combined data from the experiment without relearning (1200 runs)
* `data/experiment_results_relearn.csv` - combined data from an experiment with relearning (120 runs)

### Reproducibility of results

Specify all parameters in `main` function (`run_experiment.py` file), and run the following code: 

```python
pip install -r requirements.txt
python3 run_experiment.py
```

Although it's a bad practice to make changes in code to run a new experiment, 
the project is rather simple, so no config files were used. 