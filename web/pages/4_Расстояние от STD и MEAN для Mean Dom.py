import streamlit as st
from utils import get_dataframe

import pandas as pd
import plotly.graph_objects as go

dataframe = get_dataframe()
st.title('Расстояние от STD и MEAN для *Mean Dom*')
st.write('График расстояния от STD (стандартного отклонения) и MEAN (среднего значения) для *Mean Dom*')
fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        gauge={'shape': "bullet"},
        delta={'reference': 0.08},
        value=0.06,
        domain={'x': [0.1, 1], 'y': [0.2, 0.9]}
))
#fig.update_layout(title_text='Расстояние от STD (стандартное отклонение) и MEAN (среднее значение) для *Mean Dom*', title_x=0.4)
st.plotly_chart(fig)