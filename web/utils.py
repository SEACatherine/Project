import inspect
import textwrap
import pandas as pd

import streamlit as st


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

@st.cache_data
def get_dataframe():
    # чтение датасета
    dataframe = pd.read_csv ("https://raw.githubusercontent.com/SEACatherine/Project/master/voice.csv")
    # приведем названия колонок к нижнему регистру и удалим знаки препинания
    dataframe.columns = dataframe.columns.str.lower().str.replace('[^\w\s]', '', regex=True)
    # преобразуем строковые метки в числовые значения для колонки 'label': male = 1, female = 0
    dict = {'label':{'male':1,'female':0}}
    dataframe.replace(dict,inplace = True)
    x = dataframe.loc[:, dataframe.columns != 'label']
    y = dataframe.loc[:,'label']

    return dataframe