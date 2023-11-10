import streamlit as st
from utils import get_dataframe

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score

dataframe = get_dataframe()

# разделим данные с использованием Repeated Stratified K-Fold
X = dataframe.drop('label',axis=1)
y = dataframe[['label']]
rskf = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=42)
lst_accu_stratified = []
for train_index, test_index in rskf.split(X, y):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# переменная input_shape
input_shape = 20
input_shape = (X.shape[1],)

# определим функцию create_model, которая создает нейронную сеть
def create_model(input_shape):
    model = Sequential()
    # входной слой
    model.add(Dense(256, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.3))
    # скрытые слои
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # выходной слой
    model.add(Dense(1, activation="sigmoid"))
    # компиляция модели
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

# переменная model
model = create_model(input_shape)

# класс callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.94):
            print("\nReached 94% accuracy so cancelling training!")
            self.model.stop_training = True

# переменная callbacks
callbacks = myCallback()

# обучаем модель нейронной сети на train наборе данных
# определим batch_size и количество эпох обучения
batch_size = 64
epochs = 100
# обучаем модель на train наборе данных
model.fit(x_train, y_train, epochs=epochs,
          batch_size=batch_size,
          callbacks=[callbacks])

# сохраним модель
model.save('model.h5')

# загружаем веса
model.load_weights('model.h5')

# цикл для предсказаний
preds = []
for i in range(0,len(x_test)):
    preds.append(model.predict(x_test)[i][0])
predictions = [1 if val >0.5 else 0 for val in preds]

# оценка производительности нейронной модели на test наборе данных
st.title('Оценка производительности нейронной модели на test наборе данных')
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
st.write(f"Evaluating the neural network model using {len(x_test)} samples...")
st.write(f"Loss: {loss:.4f}")
st.write(f"Accuracy: {accuracy*100:.2f}%")

# результат бинарных предсказаний
st.title('Результат бинарных предсказаний нейронной модели')
# точность бинарных предсказаний
st.write("Overall Accuracy Score is : {}".format(accuracy_score(y_test, predictions)))