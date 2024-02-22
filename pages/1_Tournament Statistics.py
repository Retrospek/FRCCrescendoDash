import streamlit as st
import Calc_Web as cw
import pandas as pd

test = "app.csv"
any = "2024-Crescendo-FRC-2468\\data.csv"

st.set_page_config(page_title="Tournament Statistics and Predictions", page_icon="tropy")

data = cw.get_clean_data(csv= test)
team_stats = cw.team_desc(Data=data)
bubble = cw.plot(team_stats=team_stats)
#cw.auto_paths(data)

st.markdown("# Tournament Statistics/Predictions")

st.write('''This page is dedicated to analyzing teams through match predictions satatistic based calculations, 
         and other methods of choosing the right alliance partner''')

st.plotly_chart(bubble)

st.write('''Using this bubble plot we can see that teams that tend to have more wins for total games have a higher average amp scored ''')





