from urllib.error import URLError

import altair as alt
import pandas as pd

import streamlit as st
from utils import get_dataframe

dataframe = get_dataframe()
# приведем названия колонок к нижнему регистру и удалим знаки препинания
dataframe.columns = dataframe.columns.str.lower().str.replace('[^\w\s]', '', regex=True)
# преобразуем строковые метки в числовые значения для колонки 'label': male = 1, female = 0
dict = {'label':{'male':1,'female':0}}
dataframe.replace(dict,inplace = True)
x = dataframe.loc[:, dataframe.columns != 'label']
y = dataframe.loc[:,'label']

st.table(dataframe)

chart = (
    alt.Chart(dataframe)
        .mark_bar()
        .encode(
        alt.X('label', bin=True),
         y='count()',
        )
)
