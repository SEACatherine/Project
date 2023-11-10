import streamlit as st
from utils import get_dataframe

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

dataframe = get_dataframe()
st.title('Средняя частота')
st.write('Распределение средней частоты для "meanfun"')
fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
sns.distplot(dataframe['meanfun'], kde=False, bins=30, ax=ax)
values = np.array([rec.get_height() for rec in ax.patches])
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.jet(norm(values))
for rec, col in zip(ax.patches, colors):
    rec.set_color(col)
plt.title('Distribution of Mean Frequence', size=20, color='black')
st.pyplot(fig)