import streamlit as st
import Calc_Web as cw
import pandas as pd


test = "WACO_2024.csv"
any = "Comp.csv"


st.set_page_config(page_title="Team Tele Trends",
                   page_icon="tropy",
                   layout='wide')

events = ['WACO_2024.csv', 'Custom']

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
    st.write(Data_Choice)
    st.session_state.data = Data_Choice
    try:
        data = cw.get_clean_data(pd.read_csv(st.session_state.data, on_bad_lines='skip'))
        team_stats = cw.team_desc(Data=data)
    except:
        st.error("Not Custom")
try:
    data = cw.get_clean_data(pd.read_csv(st.session_state.data, on_bad_lines='skip'))
    team_stats = cw.team_desc(Data=data)

    teams = st.sidebar.multiselect("Team Auto Trends: ", team_stats['Team Number'].unique())

    auto_amp = cw.tele_amp(data, teams=teams)
    auto_speaker = cw.tele_speaker(data, teams=teams)
    col1, col2 = st.columns(2)
    st.header("Change in Amps in Auto")
    st.plotly_chart(auto_amp)
    st.header("Change in Speaker in Auto")
    st.plotly_chart(auto_speaker)
except:
    pass
