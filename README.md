# <ins> Split Conformal Prediction under Data Contamination - Synthetic Logistic and Hypercube only</ins>
## This code has been adapted for use by [reinertdaniel](https://github.com/reinertdaniel/), all code attribution goes to [jase-clarkson](https://github.com/jase-clarkson)
This repository contains code to re-produce the synthetic logistic and hypercube, plots and tables presented within the paper. Changes have been made from the original paper for reproducibility and compatibility on modern systems.

Please direct any queries surrounding this code to [reinertdaniel](https://github.com/reinertdaniel/)

This code was developed using Python 3.9.12. Please start by installing the required packages using the ```requirements.txt``` file.  

## Classification
### Synthetic Data
To reproduce the results with the synthetic data, use the command

```
python3 main.py -c {config_file}
```
The names of the config files can be found in the ```experiment_configs/``` directory.

To reproduce the results presented across different types of classifiers, run
```
python3 main.py -c logistic_by_model.yaml
python3 main.py -c hypercube_by_model.yaml
```
Then use the notebook ```synthetic_by_model_table.ipynb``` to generate the table.

For both the table and the plots, you will need to copy and paste the name of your run into the notebook, which can be found in the 
```results/``` directory. 

