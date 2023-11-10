import streamlit as st
from utils import get_dataframe

import pandas as pd
import plotly.graph_objects as go

dataframe = get_dataframe()
st.title('Median graph')
st.write('График медианы для диапазона доминирующей частоты в акустическом сигнале')
fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=4.94,
        title={'text': 'медиана для диапазона доминирующей частоты в акустическом сигнале'},
        domain={'x': [0, 1], 'y': [0, 1]}
))
st.plotly_chart(fig)