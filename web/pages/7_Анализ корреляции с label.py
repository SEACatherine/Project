import streamlit as st
from utils import get_dataframe

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dataframe = get_dataframe()
st.title('Анализ корреляции с "label"')
st.write('Столбцы имеющие наибольшую корреляцию с колонкой "label"')
def target_coeff(dataframe, target):
    data = dataframe.corr()[target].sort_values(ascending=False)
    indices = data.index
    labels = []
    corr = []
    for i in range(1, len(indices)):
        labels.append(indices[i])
        corr.append(data[i])
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    sns.barplot(x=corr, y=labels, palette="RdBu", ax=ax)
    plt.title(f'Correlation Coefficient for {target.upper()} column')
    st.pyplot(fig)
target_coeff(dataframe, 'label')