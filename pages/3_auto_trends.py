import streamlit as st
import Calc_Web as cw

test = "St_Scouting\\app.csv"


st.set_page_config(page_title="Team Tele Trends", page_icon="tropy")

data = cw.get_clean_data(csv= test)
team_stats = cw.team_desc(Data=data)

teams = st.sidebar.multiselect("Team Auto Trends: ", team_stats['Team Number'].unique())

auto_amp = cw.tele_amp(data, teams=teams)
auto_speaker = cw.tele_speaker(data, teams=teams)
col1, col2 = st.columns(2)
st.header("Change in Amps in Auto")
st.plotly_chart(auto_amp)
st.header("Change in Speaker in Auto")
st.plotly_chart(auto_speaker)