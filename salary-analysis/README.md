## Data
- `census_income_learn.csv`: the training set
- `census_income_test.csv`: the testing set
- `census_income_metadata.txt`: information about the data
- `census_income_metadata.pdf`: additional detailed information about the data

## Project
Predict whether an individual earns more or less than \\$50K based on a number of features

## Libraries used
- `pandas` for data cleaning and manipulation
- `numpy` for mathematical operations
- `matplotlib` and `seaborn` for data visualization
- `sklearn` for machine learning
- `imblearn` to deal with dataset imbalance
- `time` to time the performance of the code
- `warnings` for aesthetic removal of warnings in the notebook

## Modules
Modules created for the purpose of this project can be found in the `helpers` folder:
- `clean.py` contains functions to facilitate the cleaning of the dataset
- `eda.py` contains functions to facilitate visual EDA of the dataset
- `model.py` contains functions to facilitate modeling on the dataset
- `evaluate.py` contains functions to facilitate evaluation of the models trained on the dataset

## Notebooks
1. `numerical_EDA.ipynb`: The numerical EDA details the first numerical exploration of the dataset. The steps taken in this notebook have been abstracted in a `clean_df()` function belonging to the `helpers.clean` module.
2. `visual_EDA.ipynb`: The visual EDA builds on top of the steps in `numerical_EDA`. It performs a visual exploration and comes up with a list of additional columns to drop.
3. `model.ipynb`: The modeling notebook starts with a dataset cleaned according to the findings from both numerical and visual EDA and preprocesses the data before training various models and improving them.

## Transformations summary
- 79,785 observations removed from the train set.
- 39,607 observations removed from the test set.
- 30 columns removed from both the train set and the test set.
- Obtained dummy values
- Scaled the data
- Split into training and testing sets
- Try SMOTE
- Try ADASYN

## Modeling
### Metric
Dataset shows imbalance, so F1 is a better metric.

### Modeling
We eliminated a few models that didn't work great.