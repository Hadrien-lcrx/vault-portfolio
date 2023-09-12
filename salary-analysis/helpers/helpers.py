import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, \
                            classification_report, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            RocCurveDisplay, \
                            f1_score, \
                            make_scorer, \
                            recall_score, \
                            roc_curve, \
                            precision_score

from sklearn.model_selection import cross_val_score, \
                                    train_test_split, \
                                    GridSearchCV

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
                         type_of_test_set: str = 'test'):
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
    print(f'The training set has {train.shape[0]} observations and  {train.shape[1]} columns.')
    print(f'The training set has {test.shape[0]} observations and  {test.shape[1]} columns.')
    
# EDA
def plot_imbalance(df: pd.dataFrame):
    fig = df.taxable_income_amount.value_counts() \
                            .reset_index()  \
                            .plot(kind='bar',
                                  x='index',
                                  y='taxable_income_amount',
                                  rot=0,
                                  color=teal,
                                  width=0.8, title='Dataset imbalance') \
                            .legend(loc='upper right')
    
    return fig

def prepare_with_focus(df: pd.DataFrame,
                       focus: str,
                       function: str = 'sum'):
    df = train.groupby(by=['taxable_income_amount', focus], as_index=False, sort=True) \
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
    
def plot_barplot(labels: np.ndarray,
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

# Modeling 

def get_model_results(train_x: np.ndarray,
                      train_y: pd.Series,
                      test_x: np.ndarray,
                      test_y: pd.Series,
                      model,
                      return_dict: bool = True):
    fit_start = time.time()
    model.fit(train_x, train_y)
    fit_end = time.time()
    predict_start = time.time()
    prediction = model.predict(test_x)
    predict_end = time.time()
    
    accuracy = accuracy_score(test_y, prediction)
    f1 = f1_score(test_y, prediction)
    recall = recall_score(test_y, prediction)
    time_to_train = round(fit_end - fit_start)
    time_to_predict = round(predict_end - predict_start)
    total_time = time_to_train + time_to_predict
    print('')
    print('Time to train:', time_to_train)
    print('Time to predict:', time_to_predict)
    print('Total time:', total_time)
    print('Accuracy:)', accuracy)
    print('F-score:', f1)
    print('Recall:', recall)
    print('')
    print(classification_report(test_y, prediction))
    print(confusion_matrix(test_y, prediction))
    print('')

    if return_dict:
        output = {'time_to_train': time_to_train,
                  'time_to_predict': time_to_predict,
                  'total_time': total_time,
                  'accuracy': accuracy,
                  'f1': f1,
                  'recall': recall}
        return prediction, output
    
    else:
        return prediction
            
    
def display_confusion_matrix(test_y: pd.Series,
                             predicted_y: np.ndarray,
                             title: str = '',
                             save: bool = False):
    cmp = ConfusionMatrixDisplay.from_predictions(test_y, predicted_y, cmap='winter')
    fig, ax = plt.subplots(figsize=(10,10))
    plt.title(title)
    cmp.plot(ax=ax)

    if save==True:
        save_title = title.lower().replace(' ', '_')
        plt.savefig(fname=f'./graphs/{save_title}.png', format='png', dpi=400)
    
# Performance

def time_train(model, *args):
    start_time = time.time()
    prediction = model(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time} seconds')
    return prediction

def time_test(model, *args):
    start_time = time.time()
    prediction = model(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time} seconds')
    return prediction

def get_model_metrics(metrics_df: pd.DataFrame,
                      model_name: str,
                      time_to_train: int,
                      time_to_predict: int,
                      total_time: int,
                      accuracy: float,
                      f1: float,
                      recall: float):
    
    df = pd.DataFrame.from_records([{'model': model_name,
                                    'time_to_train': time_to_train,
                                    'time_to_predict': time_to_predict,
                                    'total_time': total_time,
                                    'accuracy': accuracy,
                                    'f1': f1,
                                    'recall': recall
                                   }])
    
    metrics_df = metrics_df.append(df,
                                   ignore_index=False,
                                   verify_integrity=False,
                                   sort=False)
    
    return metrics_df

def train_and_test_model(metrics_df: pd.DataFrame,
                         train_x: np.ndarray,
                         train_y: pd.Series,
                         test_x: np.ndarray,
                         test_y: pd.Series,
                         model,
                         model_name: str):
    predicted_y, model_metrics = get_model_results(train_x, train_y, test_x, test_y, model, return_dict=True)
    title = f'Confusion Matrix - {model_name}'
    display_confusion_matrix(test_y=test_y, predicted_y=predicted_y, title=title, save=True)
    metrics_df = get_model_metrics(metrics_df,
                                   model,
                                   model_metrics['time_to_train'],
                                   model_metrics['time_to_predict'],
                                   model_metrics['total_time'],
                                   model_metrics['accuracy'],
                                   model_metrics['f1'],
                                   model_metrics['recall'])