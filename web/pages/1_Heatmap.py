import streamlit as st
from utils import get_dataframe

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dataframe = get_dataframe()
st.title('Heatmap')
st.write('Тепловая карта корреляций')
plt.figure(figsize=(15, 10), dpi=100)
sns.heatmap(dataframe.corr(), cmap="viridis", annot=True, linewidth=0.5)
st.pyplot(plt)