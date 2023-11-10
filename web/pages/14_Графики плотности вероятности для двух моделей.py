import streamlit as st
from utils import get_dataframe

import numpy as np
import os
import pandas as pd
import warnings
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import joblib
from sklearn.metrics import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
import missingno as msgn

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

dataframe = get_dataframe()

# разделим данные с использованием Repeated Stratified K-Fold
X = dataframe.drop('label',axis=1)
y = dataframe[['label']]
rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)

lst_accu_stratified = []
for train_index, test_index in rskf.split(X, y):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# загружаем обученные сохраненные модели
log_reg = joblib.load('log_reg.pkl') # модель логистической регрессии
model = tf.keras.models.load_model('model.h5') # модель нейронной сети

# кривая ROC и площадь под этой кривой (AUC) для нейронной модели
y_pred_keras = model.predict(x_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

# кривая ROC и площадь под этой кривой (AUC) для логистической модели
y_pred_lr = log_reg.predict_proba(x_test)[:, 1]
fpr_lr, tpr_lr, thresholds_rf = roc_curve(y_test, y_pred_lr)
auc_lr = auc(fpr_lr, tpr_lr)

# функция визуализации предсказаний модели и распределения вероятностей для положительных и отрицательных классов
def plot_pdf(y_pred, y_test, name=None, smooth=500):
    positives = y_pred[y_test.label == 1]
    negatives = y_pred[y_test.label == 0]
    N = positives.shape[0]
    n =10
    s = positives
    p, x = np.histogram(s, bins=n)
    x = x[:-1] + (x[1] - x[0])/2
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))

    N = negatives.shape[0]
    n = 10
    s = negatives
    p, x = np.histogram(s, bins=n)
    x = x[:-1] + (x[1] - x[0])/2
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))
    plt.xlim([0.0, 1.0])
    plt.xlabel('density')
    plt.ylabel('density')
    plt.title('PDF-{}'.format(name))
    plt.show()
    st.pyplot(plt)

st.title('Графики плотности вероятности для двух моделей')
plot_pdf(y_pred_keras, y_test, 'Нейронная модель')
plot_pdf(y_pred_lr, y_test, 'Модель логистической регрессии')
