import streamlit as st
from utils import get_dataframe

import pandas as pd
import plotly.express as px

dataframe = get_dataframe()
st.title('Распределение данных и их связь')
st.write('Распределение данных и их связь между "q25" и "q75", а также их связь с категорией "label (male, female)"')
fig = px.scatter(dataframe, x="q25", y="q75", color="label", size='median', hover_data=['label'])
st.plotly_chart(fig)