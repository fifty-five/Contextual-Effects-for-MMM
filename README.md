# Contextual-Effects-for-MMM
How to properly handle Contectual Effects in Marketing Mix Modeling

## Main information

We used ***python 3.8*** to run the notebook.

The **requirements.txt** file contains all the packages to install. 


## Files

The files **external_factors_functions.py** and **utils.py** contain some functions that will be used during the analysis



## Folders

- The folder **data** contains:
    - The input data: the csv containing the target signal, and the csv containing all the particular events dates
    - The output data: the csv file (contextual_factors.csv) contains all the decomposition of the signal through the prediction date range, and the aggragated prediction (column yhat)
- The folder **images** fill store the graph produced by the analysis notebook

## Notebook

The analysis notebook is structured in the following parts:

- Imports
- Description of the paths and global params
- Init of all the params:
    - prophet_fix_params: the prophet params that will always remain the same.
    - prophet_cv_params: the prophet params that will be able to vary during cross-validation
    - events_series_base_params: for each event name, the 4 hyperparameters base value. They can be chosen by intuition, or using the data visualisation graphs to have a sens of the plausible values.
    - event_params_cv_ranges: During cross-validation, all the combinations of these values will be tried. When one combiation is selected, all the "supp" values will be added to the base values for each hyperparameter of all the events.
- Data Visualisation: to help understand and intuit the relations between regressors and the target signal
- Cross validation over the prophet_cv_params and the event_params_cv_ranges. Selection of the best combination of hyperparameters
- For the best combination of hyperparameters: training, evaluation, prediction in the prediction date range, display of the plot and saving of the results in csv file