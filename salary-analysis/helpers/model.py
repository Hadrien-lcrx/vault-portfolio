import time

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, \
                            auc, \
                            classification_report, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            f1_score, \
                            make_scorer, \
                            recall_score, \
                            roc_curve, \
                            RocCurveDisplay, \
                            precision_score

from sklearn.model_selection import cross_val_score, \
                                    train_test_split, \
                                    GridSearchCV

from .clean import *
from .eda import *

teal = '#007791'


my_path = os.path.abspath(__file__)

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
    print(f'Time to train: {time_to_train} seconds')
    print(f'Time to predict: {time_to_predict} seconds')
    print(f'Total time: {total_time} seconds')
    print('Accuracy:)', round(accuracy, 4))
    print('F-score:', round(f1, 4))
    print('Recall:', round(recall, 4))
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
    plt.title(title)

    if save==True:
        save_title = title.lower().replace(' ', '_')
        plt.savefig(fname=f'./graphs/{save_title}.png', format='png', dpi=400)
        
def display_roc_curve(test_y: pd.Series,
                      pred_y: np.ndarray,
                      model_name: str):
#   fpr, tpr, thresholds = roc_curve(test_y, pred_y)
#   roc_auc = auc(fpr, tpr)
#   print(fpr)
#   print('')
#   print(tpr)
#   print('')
#   print(roc_auc)
    RocCurveDisplay.from_predictions(test_y,
                                     pred_y,
                                     name=model_name)
    save_title = model_name.lower().replace(' ', '_')
    plt.savefig(fname=f'./graphs/{save_title}.png', format='png', dpi=400)
    plt.show()