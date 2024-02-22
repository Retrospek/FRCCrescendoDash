import streamlit as st
import Calc_Web as cw

st.set_page_config(page_title="Team Statistics and Predictions", page_icon="tropy")

st.sidebar.header("Team Select")
test = "app.csv"
data = cw.get_clean_data(csv= test)
team_stats = cw.team_desc(Data=data)
Red_Teams = st.sidebar.multiselect("3 Red Alliance Teams",team_stats['Team Number'].unique(), max_selections=3)
Blue_Teams = st.sidebar.multiselect("3 Blue Alliance Teams",team_stats['Team Number'].unique(), max_selections=3)
st.sidebar.header("Auto Path for team")
team_auto = st.sidebar.multiselect("Teams", team_stats['Team Number'].unique())
st.sidebar.header("Max Amplified Speaker for team")
max_speaker = st.sidebar.multiselect("amp_speak for Teams", team_stats['Team Number'].unique(), max_selections=1)
try:
    Red1 = int(Red_Teams[0])
    Red2 = int(Red_Teams[1]) 
    Red3 = int(Red_Teams[2])   
    Blue1 = int(Blue_Teams[0])  
    Blue2 = int(Blue_Teams[1]) 
    Blue3 = int(Blue_Teams[2]) 
    Blue_Pred, Red_Pred = cw.match_prediction(team_stats=team_stats, Red1=Red1, Red2=Red2, Red3=Red3, Blue1=Blue1, Blue2=Blue2, Blue3=Blue3)
    st.write("# Blue Win Prediction: ", Blue_Pred)
    st.write("# Red Win Prediction: ", Red_Pred)
except:
    st.markdown("# Data for one of the teams is insufficient")

st.markdown("# Team AutoPaths")

auto_path = cw.auto_paths(Data=data, teams=team_auto)
st.plotly_chart(auto_path)

st.markdown(f"# {max_speaker} max speaker notes in a amplified")
max_spk_amped = cw.max_amp_spk(clean_data=data, team=max_speaker)
st.write(max_spk_amped)


