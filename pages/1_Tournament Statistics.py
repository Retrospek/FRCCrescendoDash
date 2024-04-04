import streamlit as st
import Calc_Web as cw
import pandas as pd
import Home as hm
test = "WACO_2024.csv"

st.set_page_config(page_title="Tournament Statistics and Predictions",
                   page_icon="tropy",
                   layout='wide')

st.title("Tournament Statistics")

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


#test = "WACO_2024.csv"
try:
   
    bubble = cw.plot(team_stats=team_stats) #plot



    st.write('''This page is dedicated to analyzing teams through match predictions satatistic based calculations, 
            and other methods of choosing the right alliance partner''')

    st.plotly_chart(bubble)

    st.write('''Using this bubble plot we can see that teams that tend to have more wins for total games have a higher average amp scored ''')

    teams = st.multiselect("Specific Team Stats", options=team_stats['Team Number'].unique())

    teamstat = team_stats[team_stats['Team Number'].isin(teams)]

    teamstat['Team Number'] = teamstat['Team Number'].astype(int)

    teamstat.drop('Score Variability', axis=1)

    st.markdown("# :blue[Specific Team Statistics]")

    st.dataframe(teamstat.style.highlight_max(axis=0))

    with st.expander("# Similar Team List"):
        most_sim_robo = st.selectbox("Team Similar to the one Provided: ",options=team_stats['Team Number'].unique())
        cw.mst_sim_rbt(most_sim_robo, team_stats)

        

    

    ############
    #ML MODEL
    ############

    

except:
  pass
