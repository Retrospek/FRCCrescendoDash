import streamlit as st
import Calc_Web as cw

test = "St_Scouting\\app.csv"


st.set_page_config(page_title="Team Tele Trends", page_icon="tropy")

data = cw.get_clean_data(csv= test)
team_stats = cw.team_desc(Data=data)

teams = st.sidebar.multiselect("Team Tele-Op Trends: ", team_stats['Team Number'].unique())

tele_amp = cw.tele_amp(data, teams=teams)
tele_speaker = cw.tele_speaker(data, teams=teams)
col1, col2 = st.columns(2)
st.header("Change in Amps in Tele-Op")
st.plotly_chart(tele_amp)
st.header("Change in Speaker in Tele-Op")
st.plotly_chart(tele_speaker)