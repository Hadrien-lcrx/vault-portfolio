import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, \
                            classification_report, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            f1_score, \
                            make_scorer, \
                            recall_score, \
                            roc_curve, \
                            precision_score

from sklearn.model_selection import cross_val_score, \
                                    train_test_split, \
                                    GridSearchCV

from .clean import *

teal = '#007791'

# EDA
def plot_imbalance(df: pd.DataFrame):
    fig = df.taxable_income_amount.value_counts() \
                                  .reset_index()  \
                                  .plot(kind='bar',
                                        x='index',
                                        y='taxable_income_amount',
                                        rot=0,
                                        color=teal,
                                        width=0.8,
                                        title='Dataset imbalance') \
                                  .legend(loc='upper right')
    
    return fig

def prepare_with_focus(df: pd.DataFrame,
                       focus: str,
                       function: str = 'sum'):
    df = df.groupby(by=['taxable_income_amount', focus],
                    as_index=False,
                    sort=True) \
           .agg(function)[['taxable_income_amount', focus, 'age']] \
           .rename(columns={'age': 'aggregated'})

    return df

def get_plotting_elements(df: pd.DataFrame,
                          focus: str):
    labels = df[focus].unique()
    low_df =  df[df.taxable_income_amount == 0]
    high_df =  df[df.taxable_income_amount == 1]
    low_missing_values = [value for value in labels if value not in low_df[focus].values]
    high_missing_values = [value for value in labels if value not in high_df[focus].values]
    print('Groups: ', labels )
    print('Missing values in low_df:', low_missing_values)
    print('Missing values in high_df:', high_missing_values)
    low_bars = low_df.aggregated.values
    high_bars = high_df.aggregated.values
    return labels, low_bars, high_bars, low_df, high_df
    
def plot_barplot(focus: str,
                 labels: np.ndarray,
                 low_bars: np.ndarray,
                 high_bars: np.ndarray,
                 rotate: int = 0):  
    width = 0.35       # the width of the bars: can also be len(x) sequence
    
    fig, ax = plt.subplots()
    
    ax.bar(labels, low_bars, width, label=0)
    ax.bar(labels, high_bars, width, bottom=low_bars,
           label=1)
    
    ax.set_ylabel('sum')
    ax.set_title(f'Sum by {focus}')
    ax.legend()
    plt.xticks(rotation = rotate)