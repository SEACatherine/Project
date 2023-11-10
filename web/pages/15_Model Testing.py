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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score
import missingno as msgn
from sklearn import linear_model
import joblib
import librosa
from sklearn.metrics import *
import argparse
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

dataframe = get_dataframe()

# загрузка данных для тестирования моделей
uploaded_file = st.file_uploader("Загрузите файл данных (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    data.columns = data.columns.str.lower().str.replace('[^\w\s]', '', regex=True)
    label_dict = {'male': 1, 'female': 0}
    data['label'] = data['label'].map(label_dict)

    model_choice = st.radio("Выберите модель для тестирования", ["Модель логистической регрессии", "Нейронная модель"])
    num_epochs = 10  # по умолчанию
    if model_choice == "Нейронная модель":
        num_epochs = st.slider("Количество эпох обучения (только для Нейронной модели)", 1, 100, 10)

    if model_choice == "Модель логистической регрессии":
        model = joblib.load('log_reg.pkl') 
    else:
        model = tf.keras.models.load_model('model.h5')

    test_button = st.button("Тестировать модель")

    if test_button:
        X = data.drop('label', axis=1)
        y = data['label']

        model_predictions = model.predict(X)

        if model_choice == "Модель логистической регрессии":
            accuracy = accuracy_score(y, model_predictions)
            precision = precision_score(y, model_predictions)
            recall = recall_score(y, model_predictions)
            f1 = f1_score(y, model_predictions)
        else:
            model_binary_predictions = [1 if val > 0.5 else 0 for val in model_predictions.flatten()]
            accuracy = accuracy_score(y, model_binary_predictions)
            precision = precision_score(y, model_binary_predictions)
            recall = recall_score(y, model_binary_predictions)
            f1 = f1_score(y, model_binary_predictions)

        st.title('Тестирование модели')
        st.header(model_choice)
        st.write(f"Точность: {accuracy * 100:.2f}%")
        st.write(f"Точность (Precision): {precision:.2f}")
        st.write(f"Полнота (Recall): {recall:.2f}")
        st.write(f"F1-мера: {f1:.2f}")

        if model_choice == "Нейронная модель":
            st.header('Нейронная модель')
            st.write(f"Количество эпох обучения: {num_epochs}")




