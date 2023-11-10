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

st.title('Визуализация производительности двух моделей')

# график ROC curve
plt.figure(1, figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Нейронная модель (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_lr, tpr_lr, label='Модель логистической регрессии (area = {:.3f})'.format(auc_lr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
st.pyplot(plt)

# график ROC curve (zoomed in at top left)
plt.figure(2, figsize=(8, 6))
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Нейронная модель (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_lr, tpr_lr, label='Модель логистической регрессии (area = {:.3f})'.format(auc_lr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
st.pyplot(plt)