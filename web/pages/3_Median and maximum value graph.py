import streamlit as st
from utils import get_dataframe

import pandas as pd
import plotly.graph_objects as go

dataframe = get_dataframe()
st.title('Median and maximum value graph')
st.write('График сравнения медианы и максимального значения для диапазона доминирующей частоты в акустическом сигнале')
fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        gauge={'shape': "bullet"},
        delta={'reference': 21.5},
        value=4.94,
        domain={'x': [0.1, 1], 'y': [0.2, 0.9]}
))
#fig.update_layout(title_text='Cравнение медианы и максимального значения для диапазона доминирующей частоты в акустическом сигнале', title_x=0.5)
st.plotly_chart(fig)