import streamlit as st
import Calc_Web as cw

st.set_page_config(page_title="Team Statistics and Predictions", page_icon="tropy")

st.sidebar.header("Team Select")
test = "app.csv"
any = "Comp.csv"

data = cw.get_clean_data(csv= test)
team_stats = cw.team_desc(Data=data)
Red_Teams = st.sidebar.multiselect("3 Red Alliance Teams",team_stats['Team Number'].unique(), max_selections=3)
Blue_Teams = st.sidebar.multiselect("3 Blue Alliance Teams",team_stats['Team Number'].unique(), max_selections=3)
try:
    Red1 = int(Red_Teams[0])
    Red2 = int(Red_Teams[1]) 
    Red3 = int(Red_Teams[2])   
    Blue1 = int(Blue_Teams[0])  
    Blue2 = int(Blue_Teams[1]) 
    Blue3 = int(Blue_Teams[2]) 
    Blue_Pred, Red_Pred = cw.match_prediction(team_stats=team_stats, Red1=Red1, Red2=Red2, Red3=Red3, Blue1=Blue1, Blue2=Blue2, Blue3=Blue3)
    col1, col2 = st.columns(2)
    col1.st.write("# Red Win Prediction: ", Red_Pred)
    col1.st.button(f'{Red1}')
    col1.st.button(f'{Red2}')
    col1.st.button(f'{Red3}')
    col2.st.write("# Blue Win Prediction: ", Blue_Pred)
    col2.st.button(f'{Blue1}')
    col2.st.button(f'{Blue2}')
    col2.st.button(f'{Blue3}')
except:
    st.markdown("# Please enter Red and Blue Alliance team numbers")






