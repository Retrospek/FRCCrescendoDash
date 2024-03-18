import streamlit as st
import Calc_Web as cw
import pandas as pd


test = "WACO_2024.csv"
any = "Comp.csv"


st.set_page_config(page_title="Team Tele Trends",
                   page_icon="tropy",
                   layout='wide')

st.title("AUTO TRENDS")

events = ['WACO_2024.csv','Fort Worth (1).csv', 'Custom']

if 'data' not in st.session_state:
    st.session_state.data = 'WACO_2024.csv'  # Set default value

# Check if the current data exists in the events list, reset if not
if st.session_state.data not in events:
    st.session_state.data = 'WACO_2024.csv'

Data_Choice = st.selectbox('DataSet:', events, index = events.index(st.session_state.data))

if Data_Choice == 'Custom':
    Data_Provided = st.file_uploader("Your DataSet")
    try:
        st.session_state.data = Data_Provided
        data = cw.get_clean_data(pd.read_csv(st.session_state.data, on_bad_lines='skip'))
        team_stats = cw.team_desc(Data=data)
    except:
        st.error("Please Enter Your Data")

else:
    st.session_state.data = Data_Choice
    try:
        data = cw.get_clean_data(pd.read_csv(st.session_state.data, on_bad_lines='skip'))
        team_stats = cw.team_desc(Data=data)
    except:
        st.error("Not Custom")
        
try:
    team_numbers = team_stats['Team Number'].unique().tolist()

    teams = st.sidebar.multiselect("Team Tele-Op Trends: ", team_numbers)
    all = st.sidebar.checkbox("All Teams")
    if len(teams) != 0:
      if all:
          teams = team_numbers
      auto_amp = cw.auto_amp(data, teams=teams)
      auto_speaker = cw.auto_speaker(data, teams=teams)
      col1, col2 = st.columns(2)
      st.header("Change in Amps in Auto")
      st.plotly_chart(auto_amp)
      st.header("Change in Speaker in Auto")
      st.plotly_chart(auto_speaker)
    else: 
      st.write("Please enter the teams you want to analyze on the left sidebar")
except:
    pass
