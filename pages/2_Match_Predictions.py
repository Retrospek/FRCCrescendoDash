import streamlit as st
import Calc_Web as cw
import pandas as pd
import sklearn as skl


# Specify the path to the module (1_page.py in this case)
st.set_page_config(page_title="Team Statistics and Predictions",
                   page_icon="tropy",
                   layout='wide')


st.title("Team Statistics")


events = ['WACO_2024.csv','Fort Worth (1).csv', 'State.csv', 'CMP.csv','Custom']

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
    except Exception as error:
        st.error("Not Custom")

st.write("""Use a dataset specific to a competition so past results don't skew future results""")
# Get your matches into dataframes
event_matches = ['waco_matches.txt', 'fort_worth_matches.txt']
matches = cw.get_matches_cleaned(event_matches) #get all the match txt files into a dataframe format inside of a list     
#-------
txchamps_matches = cw.get_matches_cleaned(['tx_champs.txt'])

st.sidebar.header("Team Select")
#test = "WACO_2024.csv"
try:
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
        st.header("Statistics Based Predictions")
        st.write("Red Win Prediction: ", Red_Pred)
        st.write("Blue Win Prediction: ", Blue_Pred)
        if Red_Pred > Blue_Pred:
            st.write("Prediction: :red[RED]")
        elif Red_Pred < Blue_Pred:
            st.write("Prediction: :blue[BLUE]")
        #st.write(data.columns)
        #st.dataframe(data.loc[data['team_#'] == 2468][['tele_pass_source', 'tele_pass_midfield', 'tele_spk_scored', 'tele_amp_scored']])
        #
        #statistics_accur = cw.test_stats_model(txchamps_matches, team_stats)
        #st.write("Statistics Accuracy: ", statistics_accur)
        waco_data = cw.get_clean_data(pd.read_csv('WACO_2024.csv', on_bad_lines='skip'))
        waco_team_stats = cw.team_desc(Data=waco_data)
        #cw.ml_melody_model(team_stats=waco_team_stats)
        #cw.melody_rp(Red1, Red2, Red3, Blue1, Blue2, Blue3, team_stats)
        ############# MACHINE LEARNING PORTION ##########################
        # Find all the events not the custom ones => want to pass this into the ml model as multiple tournaments and matches
        events = [x for x in events if x != 'Custom']
        #-------
        ### Gonna list some useful vars ######### Meant for user data not ml data
        #team_stats # averages and st.dev of specific teams in a tourney from a csv provided by user
        #Red_Teams # List of red teams that are in a match
        #Blue_Teams  # List of blue teams that are in a match
        #-------
        # Let's create our ml statistic dataframes that has the attributes that I care about for the model
#        ml_team_stats = cw.ml_clean(events) #Stores the team_statistics for each tournament in a list for the ml data
                    #training_data = cw.get_clean_data(pd.read_csv('combine.csv', on_bad_lines='skip'))
                    #training_stats = cw.team_desc(training_data)
        #-------
        # ALL TOURNMENT ACCURACY
#        x_train, y_train = cw.ml_data(matches, ml_team_stats)
#        accur_ml = cw.ml_model(X_TRAIN=x_train, Y_TRAIN=y_train )
        #accur_nn = cw.neural_net(X_TRAIN=x_train, Y_TRAIN=y_train)
#        user_stats = cw.ml_clean([st.session_state.data])
#        cw.use_model(Red_Teams, Blue_Teams,user_stats, accur_ml)#, accur_nn
#        st.write("Beware lack of data may lead to underrepresentation for a robot, so reference both models")
#        st.write(""" Once a ample amount of data is acquired you can prioritize both of the the machine learning models""")
        
        
        # WACO ACCURACY
        waco_team_stats = cw.ml_clean(['WACO_2024.csv'])
        waco_matches = ['waco_matches.txt']
        w_matches = cw.get_matches_cleaned(waco_matches)
        X_train, Y_train = cw.ml_data(w_matches, waco_team_stats)
        waco_accur = cw.test_model(X_train,Y_train)
        st.write(":green[WACO Accuracy]", waco_accur * 100)

        # Fort Worth ACCURACY
        fw_team_stats = cw.ml_clean(['Fort Worth (1).csv'])
        fort_matches = ['fort_worth_matches.txt']
        fw_matches = cw.get_matches_cleaned(fort_matches)
        X_train, Y_train = cw.ml_data(fw_matches, fw_team_stats)
        fw_accur = cw.test_model(X_train,Y_train)
        st.write(":green[Fort Worth Accuracy]", fw_accur * 100)
    
        #Champs Accuracy
        #tx_champs_stats = cw.ml_clean(['State.csv'])
        #chmp_matches = ['tx_champs.txt']
        #champ_matches = cw.get_matches_cleaned(chmp_matches)
        #X_train, Y_train = cw.ml_data(champ_matches, tx_champs_stats)
        #chmp_accur = cw.test_model(X_train,Y_train)
        #st.write(":green[STECHMP Accuracy]", chmp_accur * 100)
        #Predictive of all stats
        #cw.ml_melody_model(team_stats=team_stats)
        #cw.use_melody_model(Red1,Red2,Red3,Blue1,Blue2,Blue3,team_stats=team_stats)

        #New Model
        ml_team_stats = cw.ml_clean(['CMP.csv']) #Stores the team_statistics for each tournament in a list for the ml data
                    #training_data = cw.get_clean_data(pd.read_csv('combine.csv', on_bad_lines='skip'))
                    #training_stats = cw.team_desc(training_data)
        #-------
        # ALL TOURNMENT ACCURACY
        matches = cw.get_matches_cleaned(['w_chp_matches.txt'])
        x_train, y_train = cw.ml_data(matches, ml_team_stats)
        accur_ml = cw.ml_model(X_TRAIN=x_train, Y_TRAIN=y_train )
        #accur_nn = cw.neural_net(X_TRAIN=x_train, Y_TRAIN=y_train)
        user_stats = cw.ml_clean([st.session_state.data])
        cw.use_model(Red_Teams, Blue_Teams,user_stats, accur_ml)#, accur_nn

        #World Champs Accuracy
        w_champs_stats = cw.ml_clean(['CMP.csv'])
        wchmp_matches = ['w_chp_matches.txt']
        wchamp_matches = cw.get_matches_cleaned(wchmp_matches)
        #st.write(w_champs_stats[0])
        #st.write(wchamp_matches)
        X_train, Y_train = cw.ml_data(wchamp_matches, w_champs_stats)
        wchmp_accur = cw.test_model(X_train,Y_train)
        st.write(":green[WCHMP Accuracy]", wchmp_accur * 100)

    except Exception as error:
      st.write(error)
      st.write("""# Please enter 3 teams for the red alliance, and 3 teams for the blue alliance to get a prediction.""")
        
    

except Exception as error:
    pass



