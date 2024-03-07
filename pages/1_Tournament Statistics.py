import streamlit as st
import Calc_Web as cw
import pandas as pd
import Home as hm
test = "WACO_2024.csv"

st.set_page_config(page_title="Tournament Statistics and Predictions",
                   page_icon="tropy",
                   layout='wide')

st.title("Tournament Statistics")

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



st.sidebar.header("Team Select")
#test = "WACO_2024.csv"
try:

    data = cw.get_clean_data(pd.read_csv(st.session_state.data, on_bad_lines='skip'))

    team_stats = cw.team_desc(Data=data) #Get team_stats from clean data

    bubble = cw.plot(team_stats=team_stats) #plot



    st.write('''This page is dedicated to analyzing teams through match predictions satatistic based calculations, 
            and other methods of choosing the right alliance partner''')

    st.plotly_chart(bubble)

    st.write('''Using this bubble plot we can see that teams that tend to have more wins for total games have a higher average amp scored ''')

    teams = st.multiselect("Specific Team Stats", options=team_stats['Team Number'].unique())

    teamstat = team_stats[team_stats['Team Number'].isin(teams)]

    st.dataframe(teamstat)
except:
    pass


