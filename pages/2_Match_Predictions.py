import streamlit as st
import Calc_Web as cw
import pandas as pd
import sklearn as skl


# Specify the path to the module (1_page.py in this case)
st.set_page_config(page_title="Team Statistics and Predictions",
                   page_icon="tropy",
                   layout='wide')


st.title("Team Statistics")


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

st.write("""Recomendation: Write a script that combines previous datasets, so you don't have to depend on new and volatile data that may skew
         your results. => Leads to a higher accuracy""")

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
        ml_team_stats = cw.ml_clean(events) #Stores the team_statistics for each tournament in a list for the ml data
                    #training_data = cw.get_clean_data(pd.read_csv('combine.csv', on_bad_lines='skip'))
                    #training_stats = cw.team_desc(training_data)
        #-------
        # Get your matches into dataframes
        event_matches = ['waco_matches.txt', 'fort_worth_matches.txt']
        matches = cw.get_matches_cleaned(event_matches) #get all the match txt files into a dataframe format inside of a list     
        #---
        x_train, y_train = cw.ml_data(matches, ml_team_stats)
        cw.ml_model(X_TRAIN=x_train, Y_TRAIN=y_train )
        #cw.neural_net(X_TRAIN=x_train, Y_TRAIN=y_train)
        user_stats = cw.ml_clean([st.session_state.data])
        cw.use_model(Red_Teams, Blue_Teams, user_stats)
        st.write("Beware lack of data may lead to underrepresentation for a robot, so reference both models")
        st.write(""" Once a ample amount of data is acquired you can prioritize the machine learning model""")

    except:
      st.write("""# Please enter 3 teams for the red alliance, and 3 teams for the blue alliance to get a prediction.""")
        
    

except:
    pass


