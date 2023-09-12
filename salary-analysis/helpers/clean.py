import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

teal = '#007791'

# Cleaning

def clean_df(df: pd.DataFrame,
             *args):
    # Change taxable_income to 0 and 1
    boolean_map = {'- 50000.': 0, 
                   '50000+.': 1}
    
    df.taxable_income_amount.replace(boolean_map, inplace=True)
    
    # Prevent data leakage
    columns_to_drop_leaking = ['wage_per_hour', 'capital_gains',
                               'capital_losses', 'divdends_from_stocks',
                               'unknown_column']
    
    df.drop(columns=columns_to_drop_leaking, inplace=True)
    
    # Get proper missing values (and duplicate columns)
    df.replace(to_replace='Not in universe.*', value=np.nan, regex=True, inplace=True)
    df.replace(to_replace='\?', value=np.nan, regex=True, inplace=True)
    nan_proportion_by_col = df.isnull().mean()
    high_nan_columns = nan_proportion_by_col[nan_proportion_by_col > 0.30].index
    df.drop(columns=high_nan_columns, inplace=True)
    
    # Remove kids (may not keep, unless it helps get rid of cleaning values)
    df = df[df.age >= 18]
    
    # Remove not in labor (may not keep, unless it helps get rid of cleaning values)
    df = df[df.full_or_part_time_employment_stat != 'Not in labor force']
    
    if len(list(args)) > 0:
        df.drop(columns=list(args), inplace=True)
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    return df

def get_cleaning_metrics(raw_train: pd.DataFrame,
                         clean_train: pd.DataFrame,
                         raw_test: pd.DataFrame,
                         clean_test: pd.DataFrame,
                         type_of_test_set: str = 'Test'):
    train_observations_removed = raw_train.shape[0] - clean_train.shape[0] 
    test_observations_removed = raw_test.shape[0] - clean_test.shape[0]
    train_columns_removed = raw_train.shape[1] - clean_train.shape[1]
    test_columns_removed = raw_test.shape[1] - clean_test.shape[1] 
    train_observations_removed_ratio = train_observations_removed / raw_train.shape[0] * 100
    test_observations_removed_ratio = test_observations_removed / raw_test.shape[0] * 100
    train_columns_removed_ratio = train_columns_removed / raw_train.shape[1] * 100
    test_columns_removed_ratio = test_columns_removed / raw_test.shape[1] * 100

    print(f'{train_observations_removed} observations removed from the train set.')
    print(f'{test_observations_removed} observations removed from the {type_of_test_set} set.')
    print(f'{train_columns_removed} columns removed from the train set.')
    print(f'{test_columns_removed} columns removed from the {type_of_test_set} set.')
    print('')
    print(f'Train dataset observations reduced by {round(train_observations_removed_ratio, 1)}%.')
    print(f'{type_of_test_set} dataset observations reduced by {round(test_observations_removed_ratio, 1)}%.')
    print(f'Train dataset columns reduced by {round(train_columns_removed_ratio, 1)}%.')
    print(f'{type_of_test_set} dataset columns reduced by {round(test_columns_removed_ratio, 1)}%.')
    print('')
    print(f'The training set has {clean_train.shape[0]} observations and  {clean_train.shape[1]} columns.')
    print(f'The training set has {clean_test.shape[0]} observations and  {clean_test.shape[1]} columns.')