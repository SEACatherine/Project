import streamlit as st
from utils import get_dataframe

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import linear_model
import joblib

dataframe = get_dataframe()

# разделим данные с использованием Repeated Stratified K-Fold
X = dataframe.drop('label',axis=1)
y = dataframe[['label']]
rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)
lst_accu_stratified = []
for train_index, test_index in rskf.split(X, y):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# функция для модели логистической регрессии
def log_reg_with_repeat_fold(data,model):
    X = data.drop('label',axis=1)
    y = data[['label']]
    rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)
    lst_accu_stratified = []
    for train_index, test_index in rskf.split(X, y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)
        lst_accu_stratified.append(model.score(x_test, y_test))
    # сохраним модель
    filename = 'log_reg.pkl'
    joblib.dump(model, filename)
    return model, lst_accu_stratified, x_train, x_test

# загружаем модель логистической регрессии
log_reg, lst_accu_stratified, x_train, x_test = log_reg_with_repeat_fold(data=dataframe, model=linear_model.LogisticRegression())
max_accuracy = max(lst_accu_stratified) * 100
min_accuracy = min(lst_accu_stratified) * 100
overall_accuracy = np.mean(lst_accu_stratified) * 100
std_deviation = np.std(lst_accu_stratified)
st.title("Logistic Regression Model Training Results")
st.write("Model training results:")
st.write("Maximum Accuracy: {} %".format(max_accuracy))
st.write("Minimum Accuracy: {} %".format(min_accuracy))
st.write("Overall Accuracy: {} %".format(overall_accuracy))
st.write("Standard Deviation: {}".format(std_deviation))
st.write("\n*Train and Test sets are split")
st.write("Train data shape:{}".format(x_train.shape))
st.write("Test data shape:{}".format(x_test.shape))