import streamlit as st
import Calc_Web as cw
import pandas as pd
import Home as hm
test = "WACO_2024.csv"

st.set_page_config(page_title="Tournament Statistics and Predictions",
                   page_icon="tropy",
                   layout='wide')

st.title("Tournament Statistics")

events = ['WACO_2024.csv','Fort Worth (1).csv', 'State.csv', 'CMP.csv', 'Custom']

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
    st.write('''This page is dedicated to analyzing teams through match predictions satatistic based calculations, 
            and other methods of choosing the right alliance partner''')


    with st.expander(":red[Bubble Plot of Averages and Wins]"):
        bubble = cw.plot(team_stats=team_stats) #plot
        st.plotly_chart(bubble)
        st.write('''Using this bubble plot we can see that teams that tend to have more wins for total games have a higher average amp scored ''')

    

    match_info = st.sidebar.selectbox("Match Info", options=team_stats['Team Number'].unique())
    
    static_attributes = ['team_#', 'match_#', 'auto_amp_scored', 'auto_spk_scored', 'endgame_stage_actions','endgame_trap_scored', 'tele_amp_scored', 'tele_spk_scored']
    

    try:
        match_table = data.loc[data['team_#'] == int(match_info)][static_attributes + ['tele_feeded']]
    except Exception as error:
        try:
            match_table = data.loc[data['team_#'] == int(match_info)][static_attributes + ['tele_pass_source','tele_pass_midfield']]
        except Exception as error:
            pass
    
    with st.expander(f":red[Match Info for {match_info}]"):
        st.dataframe(match_table.sort_values('match_#'))


    st.markdown("# :blue[Specific Team Statistics]")

    teams = st.sidebar.multiselect("Specific Team Stats", options=team_stats['Team Number'].unique())

    teamstat = team_stats[team_stats['Team Number'].isin(teams)]

    teamstat['Team Number'] = teamstat['Team Number'].astype(int)

    teamstat.drop('Score Variability', axis=1)

    st.dataframe(teamstat.style.highlight_max(axis=0))

    #with st.expander("# :red[Similar Team List]"):
    most_sim_robo = st.selectbox("Team Similar to the one Provided: ",options=team_stats['Team Number'].unique())
    robot_categ = st.selectbox("Robot Category", ["Aggressive", "Consistent", "Feeder"])

    weights = []

    # Aggressive weights
    aggressive_weights = [0, 0.05, 0.05, 0.1, 0.2, 0.05, 0.1, 0.1, 0.2, 0.1, 0.15, 0.05, 0.05]

    # Consistent weights
    consistent_weights = [0, 0.02, 0.02, 0.05, 0.1, 0.02, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05]

    # Feeder weights
    feeder_weights = [0, 0.01, 0.02, 0.05, 0.1, 0.01, 0.03, 0.03, 0.05, 0.05, 0.05, 0.3, 0.3] 
    if robot_categ == 'Aggressive':
        weights = aggressive_weights
    elif robot_categ == 'Consistent':
        weights = consistent_weights
    elif robot_categ == 'Feeder':
        weights = feeder_weights

    cw.mst_sim_rbt(team=most_sim_robo, team_stats=team_stats, weights=weights)
        
    
        

    

    ############
    #ML MODEL
    ############

    

except Exception as error:
  st.write(error)
  pass
