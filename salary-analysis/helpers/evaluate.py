import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

teal = '#007791'

from .clean import *
from .eda import *
from .model import *

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

# Evaluate

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
    print('')
    print('')
    print(model_name.upper())
    predicted_y, model_metrics = get_model_results(train_x, train_y, test_x, test_y, model, return_dict=True)
    title = f'Confusion Matrix - {model_name}'
    display_confusion_matrix(test_y=test_y, predicted_y=predicted_y, title=title, save=True)
    display_roc_curve(test_y, predicted_y, model_name)
    metrics_df = get_model_metrics(metrics_df,
                                   model_name,
                                   model_metrics['time_to_train'],
                                   model_metrics['time_to_predict'],
                                   model_metrics['total_time'],
                                   model_metrics['accuracy'],
                                   model_metrics['f1'],
                                   model_metrics['recall'])
    return metrics_df
    