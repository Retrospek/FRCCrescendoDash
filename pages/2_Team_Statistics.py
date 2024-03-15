import streamlit as st
import Calc_Web as cw
import pandas as pd
import sklearn as skl


# Specify the path to the module (1_page.py in this case)
st.set_page_config(page_title="Team Statistics and Predictions",
                   page_icon="tropy",
                   layout='wide')


st.title("Team Statistics")


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
        
        
        st.header("Machine Learning Based Predictions")
        matches = cw.get_matches_cleaned()
        training_data = cw.get_clean_data(pd.read_csv('WACO_2024.csv', on_bad_lines='skip'))
        training_stats = cw.team_desc(training_data)
        data = cw.ml_clean(team_stats=team_stats)
        train_data = cw.ml_clean(team_stats=training_stats)
        x_train, y_train = cw.ml_data(matches, train_data)
        cw.ml_model(X_TRAIN=x_train, Y_TRAIN=y_train )

        cw.use_model(Red_Teams, Blue_Teams, data)
        st.write("Beware lack of data may lead to overfitting, so reference both models")

    except Exception as error:
        st.write(error)
        st.markdown("# Please enter Red and Blue Alliance team numbers")
    

except:
    pass




