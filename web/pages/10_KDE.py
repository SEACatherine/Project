import streamlit as st
from utils import get_dataframe

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dataframe = get_dataframe()
st.title('KDE')
st.write('KDE-графики для каждого признака, разделенные по классам (female = 0, male = 1)')
fig, axes = plt.subplots(4, 5, figsize=(20, 20))
features = dataframe.columns[:-1]  # Исключаем последний столбец 'label'
for i in range(4):
    for j in range(5):
        k = i * 5 + j + 1
        ax = axes[i, j]
        if k <= 20:
            ax.set_title(features[k - 1])
            sns.kdeplot(dataframe.loc[dataframe['label'] == 0, features[k - 1]], color='green', label='F', ax=ax)
            sns.kdeplot(dataframe.loc[dataframe['label'] == 1, features[k - 1]], color='red', label='M', ax=ax)
plt.tight_layout()
st.pyplot(fig)