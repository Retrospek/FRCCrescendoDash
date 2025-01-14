import pandas as pd
import numpy as np
import math
import re
import os
#import statbotics
import matplotlib.pyplot as plt
import random
import scipy.integrate as integrate
from numpy import sqrt, cos, sin, pi
from scipy.stats import norm
import streamlit as st
#import networkx as nx
#import more_itertools as itr
import scipy.stats as sta
import matplotlib.image as mpimg
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.io as pio
from dash import dcc
import plotly.graph_objects as go
#import sci
import base64
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

#import tensorflow as tf
#from keras .models import Sequential
#from keras .layers import Dense
#from keras.models import load_model
import joblib

#'Scouting\Team_Analysis\default (1).csv'
@st.cache_data
def get_clean_data(data):
    default = data
    attributes = ['tele_spk_scored', 'tele_amp_scored']
    if 'tele_pass_midfield' in data.columns:
        attributes.append('tele_pass_midfield')
    if 'tele_pass_source' in data.columns:
        attributes.append('tele_pass_source')
    for i in range(len(default)):
        
        for j in range(len(attributes)):
            if pd.isna(default.loc[i, attributes[j]]):# or len(default.at[i, attributes[j]]) == 0:
                default.at[i,attributes[j]] = 0
            else:
                if str(default.at[i, attributes[j]]).find('|') != -1:
                    default.loc[i, attributes[j]] = len(default.at[i, attributes[j]].split('|'))
                else:   
                    default.at[i, attributes[j]] = 1
                


            #st.write(default.loc[i, attributes[j]])
   
    dict = {None : 0}
    default.replace({"auto_amp_scored": dict})
    default.replace({"auto_amp_scored": dict})

    return default
#team_stats = pd.DataFrame(columns=['Team-Number', 'AVG_AMP_AUTO', 'AVG_SPEAKER_AUTO', 'AVG_AMP_TELE', 'AVG_SPEAKER_TELE', 'STD_AMP_AUTO', 'STD_SPEAKER_AUTO', 'STD_AMP_TELE', 'STD_SPEAKER_TELE' ])
@st.cache_data
def team_desc(Data):
    Data.dropna(subset='team_#',inplace=True) 
    team_numbers = Data['team_#'].unique()
    team_stats = pd.DataFrame(team_numbers, columns=['Team Number'])
    team_stats.insert(1,'AVG_AMP_AUTO', 0)
    team_stats.insert(2,'AVG_SPEAKER_AUTO', 0)
    team_stats.insert(3,'AVG_AMP_TELE', 0)
    team_stats.insert(4,'AVG_SPEAKER_TELE', 0)
    team_stats.insert(5,'STD_AMP_AUTO', 0)
    team_stats.insert(6,'STD_SPEAKER_AUTO', 0)
    team_stats.insert(7,'STD_AMP_TELE', 0)
    team_stats.insert(8,'STD_SPEAKER_TELE', 0)
    team_stats.insert(9, 'Predicted Score', 0)
    team_stats.insert(10, 'Score Variability', 0)
    team_stats.insert(11, 'Win/Total', 0)
    team_stats.insert(12, 'AVG_FEED_TELE', 0)
    team_stats.insert(13, 'STD_FEED_TELE', 0)



    ################# LAZZZZZZZZZZY

    #team_stats.insert(14, 'AVG_HANG_SUCCESS',0)
    #team_stats.insert(15, 'STD_HANG_SUCCESS', 0)

    for i in range(len(team_stats)):
        team = team_stats.at[i, 'Team Number']
        Team_DF = Data.loc[Data['team_#'] == team]
        result_map = {'Win': 'Yes', 'Lose': 'No', 'Tie': 'No'}
        try:
            Team_DF['win'] = Team_DF['Result'].map(result_map)
        except:
            pass
        team_wins = len(Team_DF.loc[Team_DF['win'] == 'Yes'])

        team_total = len(Team_DF)
        team_stats.loc[i, 'AVG_AMP_AUTO'] = Team_DF['auto_amp_scored'].describe()[1]
        team_stats.loc[i, 'AVG_SPEAKER_AUTO'] = Team_DF['auto_spk_scored'].describe()[1]
        team_stats.loc[i, 'AVG_AMP_TELE'] = Team_DF['tele_amp_scored'].describe()[1]
        team_stats.loc[i, 'AVG_SPEAKER_TELE'] = Team_DF['tele_spk_scored'].describe()[1]
        if 'tele_pass_midfield' in Team_DF.columns:
            team_stats.loc[i, 'AVG_FEED_TELE'] = Team_DF['tele_pass_midfield'].describe()[1] + Team_DF['tele_pass_source'].describe()[1]
            team_stats.loc[i, 'STD_FEED_TELE'] = sqrt(pow(Team_DF['tele_pass_midfield'].describe()[2],2) + pow(Team_DF['tele_pass_source'].describe()[1],2))
        else:
            team_stats.loc[i, 'AVG_FEED_TELE'] = Team_DF['tele_feeded'].describe()[1]
            team_stats.loc[i, 'STD_FEED_TELE'] = Team_DF['tele_feeded'].describe()[2]

        team_stats.loc[i, 'STD_AMP_AUTO'] = Team_DF['auto_amp_scored'].describe()[2]
        team_stats.loc[i, 'STD_SPEAKER_AUTO'] = Team_DF['auto_spk_scored'].describe()[2]
        team_stats.loc[i, 'STD_AMP_TELE'] = Team_DF['tele_amp_scored'].describe()[2]
        team_stats.loc[i, 'STD_SPEAKER_TELE'] = Team_DF['tele_spk_scored'].describe()[2]


        team_stats.loc[i, 'Win/Total'] = team_wins/team_total
    team_stats['AVG_AMP_AUTO'].fillna(0, inplace=True)
    team_stats['AVG_SPEAKER_AUTO'].fillna(0, inplace=True)
    team_stats['AVG_AMP_TELE'].fillna(0, inplace=True)
    team_stats['AVG_SPEAKER_TELE'].fillna(0, inplace=True)
    team_stats['AVG_FEED_TELE'].fillna(0, inplace=True)
    team_stats['STD_FEED_TELE'].fillna(0, inplace=True)
    team_stats['STD_AMP_AUTO'].fillna(0, inplace=True)
    team_stats['STD_SPEAKER_AUTO'].fillna(0, inplace=True)
    team_stats['STD_AMP_TELE'].fillna(0, inplace=True)
    team_stats['STD_SPEAKER_TELE'].fillna(0, inplace=True)
    team_stats['Win/Total'].fillna(0, inplace=True)
    for j in range(len(team_stats)):
        #if team_stats.at[j, 'Team Number'] == 1234:    
        team_stats.loc[j,'Predicted Score'] = team_stats.at[j, 'AVG_AMP_AUTO']*2 + team_stats.at[j, 'AVG_SPEAKER_AUTO']*5 + team_stats.at[j, 'AVG_AMP_TELE']*1 + team_stats.at[j, 'AVG_SPEAKER_TELE']*2
        team_stats.loc[j, 'Score Variability'] = math.sqrt(pow(team_stats.at[j,'STD_AMP_AUTO']*2, 2) + pow(team_stats.at[j,'STD_SPEAKER_AUTO']*5, 2) + pow(team_stats.at[j,'STD_AMP_TELE']*1, 2) + pow(team_stats.at[j,'STD_SPEAKER_TELE'] * 2, 2) )
        if pd.isna(team_stats.at[j, 'Score Variability']):
            team_stats.loc[j, 'Score Variability'] == 0
    return team_stats
 
def match_prediction(team_stats, Red1, Red2, Red3, Blue1, Blue2, Blue3):
#print("Type your 6 teams in order from 0-9")
    BlueAllianceAvg = 0
    RedAllianceAvg = 0
    BlueAllianceStd = 0
    RedAllianceStd = 0
    asked_teams = [Red1, Red2, Red3, Blue1, Blue2, Blue3]
    asked_teams_avg = [0,0,0,0,0,0]
    asked_teams_std = [0,0,0,0,0,0]
    
    for team in range(len(asked_teams)):  
        alliateam = team_stats.loc[team_stats['Team Number'] == asked_teams[team]]
        if team <= 2:
            RedAllianceAvg += alliateam['Predicted Score'].iloc[0]
            RedAllianceStd += pow(alliateam['Score Variability'].iloc[0], 2)
        else:
            BlueAllianceAvg += alliateam['Predicted Score'].iloc[0]
            BlueAllianceStd += pow(alliateam['Score Variability'].iloc[0],2)
        asked_teams_avg[team] = alliateam['Predicted Score'].iloc[0]
        asked_teams_std[team] = alliateam['Score Variability'].iloc[0]

    Alliances = pd.DataFrame([['Blue',BlueAllianceAvg, math.sqrt(BlueAllianceStd)], ['Red', RedAllianceAvg, math.sqrt(RedAllianceStd)]], columns = ['Alliance', 'Predicted', 'St.D'])

    RedAlliance = Alliances.loc[Alliances['Alliance'] == 'Red']
    BlueAlliance = Alliances.loc[Alliances['Alliance'] == 'Blue']
    mean_diff = RedAlliance['Predicted'].iloc[0] - BlueAlliance['Predicted'].iloc[0]
    std_diff = math.sqrt(pow(RedAlliance['St.D'].iloc[0], 2) + pow(RedAlliance['St.D'].iloc[0], 2))

    Z_score = (0 - mean_diff)/std_diff

    Blue_Win = norm.cdf(Z_score)


    Blue_pred = Blue_Win * 100
    Red_pred = (1-Blue_Win) * 100
    return Blue_pred, Red_pred
    #Scouting\Team_Analysis\PointCalculator.py
##################################################################################
##################################################################################
######## MACHINE LEARNING SECTION FROM HERE ON OUT ML SECTION ####################
#@st.cache_data
def ml_clean(events):
    event_stats = []
    for event in events:
        training_data = get_clean_data(pd.read_csv(event, on_bad_lines='skip'))
        training_stats = team_desc(training_data)
        training = pd.DataFrame(training_stats['Team Number'], columns=['Team Number'])
        #st.write(training_stats.columns)
        training['AUTO_AVG_SCORE'] = training_stats[['AVG_SPEAKER_AUTO', 'AVG_AMP_AUTO']].sum(axis=1)
        training['AVG_TELE_AMP'] = training_stats['AVG_AMP_TELE']
        training['AVG_TELE_SPEAKER'] = training_stats['AVG_SPEAKER_TELE']
        training['VARIABILITY'] = training_stats['Score Variability']
        #training['AUTO_STD'] = training_stats
        training['AVG_FEED_TELE'] = training_stats['AVG_FEED_TELE']
        training['STD_FEED_TELE'] = training_stats['STD_FEED_TELE']
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(training.drop('Team Number', axis=1))
        normalized_df = pd.DataFrame(normalized_data, columns=['AUTO_AVG', 'AVG_TELE_SPEAKER', 'AVG_TELE_AMP', 'VARIABILITY', 'AVG_FEED_TELE', 'STD_FEED_TELE'])
        
        full_data = pd.concat([training['Team Number'], normalized_df], axis=1) 

        event_stats.append(full_data)
    return event_stats
        #st.dataframe(X)
#@st.cache_data
def get_matches_cleaned(events):
    total_df = []
    for event in events:
        file_path = event

        matches = pd.DataFrame(columns=['Red1', 'Red2', 'Red3', 'Blue1', 'Blue2', 'Blue3', 'RedScore', 'BlueScore'])
        with open(file_path, 'r') as file:
            # Read the file line by line
            line_count = 0
            for line in file:
                #st.write(line)
                line_count += 1
                #st.write(line)
                if line_count % 2 == 0:
                    # Split the line into individual values
                    values = line.strip().split('\t')
                    #st.write(values)
                    values = [float(val) if val.replace('.', '', 1).isdigit() else None for val in values]
                    # Append the new DataFrame to the total_df
                    matches.loc[len(matches.index)] = values 

        total_df.append(matches)
        #st.dataframe(total_df[0])
    return total_df
#@st.cache_data
def ml_data(all_matches, ml_team_stats):
    x_train = []
    y_train = []
    for i in range(len(all_matches)):
        matches = all_matches[i]
        stand_teams = ml_team_stats[i]
        matches['Winner'] = 'Red'  # Assume Red alliance won initially
        matches.loc[matches['BlueScore'] > matches['RedScore'], 'Winner'] = 'Blue'  # Update to 'Blue' if Blue alliance won
        aggreg = matches.drop(['RedScore', 'BlueScore'], axis=1)

        #aggreg['Team Number'] = aggreg['Team Number'].astype('float64')
        ##### Created the match outcomes
        ##### Now to create training dataset

        # Stats model has 77%
        # put probabilities into the model's featueres
        # but if accur is .77 then model will skew to .77

        # Add auto_std and split up the variability
        # Try to add statistics probabilities
        # Consider team numbers and how 'big' they are to other team numbers

        attributes = ['AUTO_AVG', 'AVG_TELE_SPEAKER', 'AVG_TELE_AMP', 'VARIABILITY', 'AVG_FEED_TELE', 'STD_FEED_TELE']
        for i in range(len(aggreg)):

            #st.write(aggreg.at[0,'Team Number'])
            Red1 = stand_teams.loc[stand_teams['Team Number'] == aggreg.at[i,'Red1']]
            Red2 = stand_teams.loc[stand_teams['Team Number'] == aggreg.at[i,'Red2']]
            Red3 = stand_teams.loc[stand_teams['Team Number'] == aggreg.at[i,'Red3']]
            Blue1 = stand_teams.loc[stand_teams['Team Number'] == aggreg.at[i,'Blue1']]
            Blue2 = stand_teams.loc[stand_teams['Team Number'] == aggreg.at[i,'Blue2']]
            Blue3 = stand_teams.loc[stand_teams['Team Number'] == aggreg.at[i,'Blue3']]
            RedAlliance = []
            BlueAlliance = []
            for j in range(len(attributes)):
                attribute = attributes[j]
                n = 0
                if not Red1.empty and not Red2.empty and not Red3.empty and not Blue1.empty and not Blue2.empty and not Blue3.empty:
                    if attribute == 'VARIABILITY':
                        RedAlliance.append(math.sqrt(pow(Red1[attribute].iloc[0], 2)+pow(Red2[attribute].iloc[0], 2)+pow(Red3[attribute].iloc[0], 2)))
                        BlueAlliance.append(math.sqrt(pow(Blue1[attribute].iloc[0], 2)+pow(Blue2[attribute].iloc[0], 2)+pow(Blue3[attribute].iloc[0], 2)))
                    elif attribute == 'STD_FEED_TELE':
                        RedAlliance.append(math.sqrt(pow(Red1[attribute].iloc[0], 2)+pow(Red2[attribute].iloc[0], 2)+pow(Red3[attribute].iloc[0], 2)))
                        BlueAlliance.append(math.sqrt(pow(Blue1[attribute].iloc[0], 2)+pow(Blue2[attribute].iloc[0], 2)+pow(Blue3[attribute].iloc[0], 2)))
                    elif attribute != 'STD_FEED_TELE' and attribute != 'VARIABILITY':
                        RedAlliance.append(Red1[attribute].iloc[0] + Red2[attribute].iloc[0] + Red3[attribute].iloc[0])
                        BlueAlliance.append(Blue1[attribute].iloc[0] + Blue2[attribute].iloc[0] + Blue3[attribute].iloc[0])
            difference = [x - y for x, y in zip(RedAlliance, BlueAlliance)]

            #if i == 1 or i == 2 or i == 3:
            #    st.write(RedAlliance)
            result = []
            
            if aggreg.at[i, 'Winner'] == 'Red':
                result.append(0)
            else:
                result.append(1)
            x_train.append(difference)
            y_train.append(result)
    return x_train, y_train
#@st.cache_data
def ml_model(X_TRAIN, Y_TRAIN): 
    #kf = KFold(n_splits=5, shuffle=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_TRAIN, Y_TRAIN, test_size=0.2, random_state=90)#90, 61, 73 randomstate good, 
    svm = SVC(kernel='sigmoid', gamma='scale', decision_function_shape='ovr')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    st.write("SVC", accuracy_score(y_test, y_pred)*100)
    joblib.dump(svm, "goat_model.joblib")
    # Naive Bayes:
    # 
    #gnb.fit(X_train, y_train)
    #y_pred = gnb.predict(X_test)
    #st.write("GaussianNB", accuracy_score(y_train,y_pred))
    #joblib.dump(gnb, 'GNB_model.joblib')
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    
def use_model(Red_teams, Blue_teams, stats_teams, accur_ml):
    stand_teams = pd.DataFrame(columns=stats_teams[0].columns)
    for i in range(len(stats_teams)):                       
        stand_teams = pd.concat([stand_teams,stats_teams[i]], ignore_index=True)
    ml_loaded_model = joblib.load('goat_model.joblib')
    #nn_loaded_model = load_model('neuralnet.h5')
    Red1 = stand_teams.loc[stand_teams['Team Number'] == Red_teams[0]]
    Red2 = stand_teams.loc[stand_teams['Team Number'] == Red_teams[1]]
    Red3 = stand_teams.loc[stand_teams['Team Number'] == Red_teams[2]]
    Blue1 = stand_teams.loc[stand_teams['Team Number'] == Blue_teams[0]]
    Blue2 = stand_teams.loc[stand_teams['Team Number'] == Blue_teams[1]]
    Blue3 = stand_teams.loc[stand_teams['Team Number'] == Blue_teams[2]]
    attributes = ['AUTO_AVG', 'AVG_TELE_SPEAKER', 'AVG_TELE_AMP', 'VARIABILITY', 'AVG_FEED_TELE', 'STD_FEED_TELE']
    RedAlliance = []
    BlueAlliance = []
    for j in range(len(attributes)):
        attribute = attributes[j]
        if attribute == 'VARIABILITY':
            RedAlliance.append(math.sqrt(pow(Red1[attribute].iloc[0], 2)+pow(Red2[attribute].iloc[0], 2)+pow(Red3[attribute].iloc[0], 2)))
            BlueAlliance.append(math.sqrt(pow(Blue1[attribute].iloc[0], 2)+pow(Blue2[attribute].iloc[0], 2)+pow(Blue3[attribute].iloc[0], 2)))
        elif attribute == 'STD_FEED_TELE':
            RedAlliance.append(math.sqrt(pow(Red1[attribute].iloc[0], 2)+pow(Red2[attribute].iloc[0], 2)+pow(Red3[attribute].iloc[0], 2)))
            BlueAlliance.append(math.sqrt(pow(Blue1[attribute].iloc[0], 2)+pow(Blue2[attribute].iloc[0], 2)+pow(Blue3[attribute].iloc[0], 2)))
        elif attribute != 'STD_FEED_TELE' and attribute != 'VARIABILITY':
            RedAlliance.append(Red1[attribute].iloc[0] + Red2[attribute].iloc[0] + Red3[attribute].iloc[0])
            BlueAlliance.append(Blue1[attribute].iloc[0] + Blue2[attribute].iloc[0] + Blue3[attribute].iloc[0])
    difference = [x - y for x, y in zip(RedAlliance, BlueAlliance)]

    x_test = []
    x_test.append(difference)

    match_pred = ml_loaded_model.predict(x_test)
    #nn_loaded_model = load_model('neuralnet.h5')
    st.header("Gradient Boosted Output")
    #st.write(f"All Tournament Accuracy: :green[{accur_ml * 100}%]")
    #st.markdown("All Tournament Accuracy: :green[89.34%]")
    if match_pred == 0:
        st.write('Prediction: :red[RED]') 
    else:
        st.write('Prediction: :blue[BLUE]')
#@st.cache_data
def test_model(X_TRAIN, Y_TRAIN):
    #st.write(X_TRAIN)
    #st.write(Y_TRAIN)
    model = joblib.load('goat_model.joblib')
    prediction = model.predict(X_TRAIN)
    accuracy = accuracy_score(Y_TRAIN, prediction)
    #st.write("testing model", accuracy)
    #model_2 = joblib.load('GNB_model.joblib')
    #prediction_2 = model_2.predict(X_TRAIN)
    #accuracy_2 = accuracy_score(Y_TRAIN, prediction_2)
    #st.write("GNB_model accur", accuracy_2)
    return accuracy
#@st.cache_data
def test_stats_model(matches, team_stats):
    successes = 0
    waco = matches[0]
    n = len(waco)
    for i in range(len(waco)):
        red1 = int(waco.iloc[i]['Red1'])
        red2 = waco.iloc[i]['Red2']
        red3 = waco.iloc[i]['Red3']
        blue1 = waco.iloc[i]['Blue1']
        blue2 = waco.iloc[i]['Blue2']
        blue3 = waco.iloc[i]['Blue3']
        RedScore = waco.iloc[i]['RedScore']
        BlueScore = waco.iloc[i]['BlueScore']

        winner = 'red' if RedScore > BlueScore else 'blue'
        blue, red = match_prediction(team_stats, red1,red2,red3,blue1,blue2,blue3)
        prediction = 'red' if red > blue else 'blue'
        if winner ==  prediction:
            successes += 1
    return successes/n

#@st.cache_data
def ml_melod_match_data(team_stats):
    file = 'waco_new_matches.txt'
    tourney = pd.DataFrame(columns=['Red1', 'Red2', 'Red3', 'Blue1', 'Blue2', 'Blue3', 'RedScore', 'BlueScore', 're', 'rm', 'be', 'bm'])
    with open(file=file,mode='r') as f:
        line_count = 0
        rows = []
        for line in f:
            line_count +=1
            if line_count %2 == 0:
                values = re.findall(r'\d+', line)
                numbers = [float(value) for value in values if value.replace('.', '', 1).isdigit()]
                # Append the new DataFrame to the total_df
                if len(numbers) == len(tourney.columns):
                    # Append the numbers as a new row to the DataFrame
                    tourney.loc[len(tourney)] = numbers
    return tourney

#@st.cache_data
def ml_melody_model(team_stats):
    matches = ml_melod_match_data(team_stats=team_stats)
    train_input = []
    train_output = []
    for i in range(len(matches)):
        row = matches.iloc[i]
        Red1 = team_stats.loc[team_stats['Team Number'] == row['Red1']]
        Red2 = team_stats.loc[team_stats['Team Number'] == row['Red2']]
        Red3 = team_stats.loc[team_stats['Team Number'] == row['Red3']]
        Blue1 = team_stats.loc[team_stats['Team Number'] == row['Blue1']]
        Blue2 = team_stats.loc[team_stats['Team Number'] == row['Blue1']]
        Blue3 = team_stats.loc[team_stats['Team Number'] == row['Blue1']]
        rm_outcome = row['rm']
        bm_outcome = row['bm']
        train_output.append(rm_outcome)
        train_output.append(bm_outcome)
        
        red_spk_avg = Red1['AVG_SPEAKER_TELE'].iloc[0] + Red2['AVG_SPEAKER_TELE'].iloc[0] + Red3['AVG_SPEAKER_TELE'].iloc[0]
        red_amp_avg = Red1['AVG_AMP_TELE'].iloc[0] + Red2['AVG_AMP_TELE'].iloc[0] + Red3['AVG_AMP_TELE'].iloc[0]
        red_avg_notes = float(red_spk_avg + red_amp_avg)
        red_notes_std = float(math.sqrt(pow(Red1['STD_AMP_TELE'].iloc[0],2) + pow(Red2['STD_AMP_TELE'].iloc[0],2) + pow(Red3['STD_AMP_TELE'].iloc[0],2) + pow(Red1['STD_SPEAKER_TELE'].iloc[0],2) + pow(Red2['STD_SPEAKER_TELE'].iloc[0],2) + pow(Red3['STD_SPEAKER_TELE'].iloc[0],2)))    
        train_input.append([red_avg_notes, red_notes_std])
        blue_spk_avg = Blue1['AVG_SPEAKER_TELE'].iloc[0] + Blue2['AVG_SPEAKER_TELE'].iloc[0] + Blue2['AVG_SPEAKER_TELE'].iloc[0]
        blue_amp_avg = Blue1['AVG_AMP_TELE'].iloc[0] + Blue2['AVG_AMP_TELE'].iloc[0] + Blue2['AVG_AMP_TELE'].iloc[0]
        blue_avg_notes = float(blue_spk_avg + blue_amp_avg)
        blue_notes_std = float(math.sqrt(pow(Blue1['STD_AMP_TELE'].iloc[0],2) + pow(Blue2['STD_AMP_TELE'].iloc[0],2) + pow(Blue3['STD_AMP_TELE'].iloc[0],2) + pow(Blue1['STD_SPEAKER_TELE'].iloc[0],2) + pow(Blue2['STD_SPEAKER_TELE'].iloc[0],2) + pow(Blue3['STD_SPEAKER_TELE'].iloc[0],2)))
        train_input.append([blue_avg_notes, blue_notes_std])
    X_train, X_test, Y_train, Y_test = train_test_split(train_input, train_output, test_size=0.2)
    
    max_accur = 0
    clf = GradientBoostingClassifier(n_estimators=11, learning_rate=0.27, random_state=42)

    #metric='euclidean', n_neighbors=8, weights='uniform'
    #weights='distance', n_neighbors=4

    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    accur = accuracy_score(Y_test, y_pred)
        
    joblib.dump(clf, 'ml_melody_model.joblib')

    check_pred = clf.predict(X_test)
    check_accur = accuracy_score(Y_test, check_pred) 
    st.write(f'Melody RP Accuracy :blue[{check_accur}]')
    
    
    """grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=1)

    grid_search.fit(X_train, Y_train)"""

    #st.write(grid_search.score(X_test, Y_test))
    #st.write(grid_search.best_params_)

def use_melody_model(red1, red2, red3, blue1, blue2, blue3, team_stats):
    model = joblib.load('ml_melody_model.joblib')
    red_match = []
    blue_match = []
    Red1 = team_stats.loc[team_stats['Team Number'] == red1]
    Red2 = team_stats.loc[team_stats['Team Number'] == red2]
    Red3 = team_stats.loc[team_stats['Team Number'] == red3]
    Blue1 = team_stats.loc[team_stats['Team Number'] == blue1]
    Blue2 = team_stats.loc[team_stats['Team Number'] == blue2]
    Blue3 = team_stats.loc[team_stats['Team Number'] == blue3]
    red_spk_avg = Red1['AVG_SPEAKER_TELE'].iloc[0] + Red2['AVG_SPEAKER_TELE'].iloc[0] + Red3['AVG_SPEAKER_TELE'].iloc[0]
    red_amp_avg = Red1['AVG_AMP_TELE'].iloc[0] + Red2['AVG_AMP_TELE'].iloc[0] + Red3['AVG_AMP_TELE'].iloc[0]
    red_avg_notes = float(red_spk_avg + red_amp_avg)
    red_notes_std = float(math.sqrt(pow(Red1['STD_AMP_TELE'].iloc[0],2) + pow(Red2['STD_AMP_TELE'].iloc[0],2) + pow(Red3['STD_AMP_TELE'].iloc[0],2) + pow(Red1['STD_SPEAKER_TELE'].iloc[0],2) + pow(Red2['STD_SPEAKER_TELE'].iloc[0],2) + pow(Red3['STD_SPEAKER_TELE'].iloc[0],2)))    
    red_match.append([red_avg_notes, red_notes_std])
    blue_spk_avg = Blue1['AVG_SPEAKER_TELE'].iloc[0] + Blue2['AVG_SPEAKER_TELE'].iloc[0] + Blue2['AVG_SPEAKER_TELE'].iloc[0]
    blue_amp_avg = Blue1['AVG_AMP_TELE'].iloc[0] + Blue2['AVG_AMP_TELE'].iloc[0] + Blue2['AVG_AMP_TELE'].iloc[0]
    blue_avg_notes = float(blue_spk_avg + blue_amp_avg)
    blue_notes_std = float(math.sqrt(pow(Blue1['STD_AMP_TELE'].iloc[0],2) + pow(Blue2['STD_AMP_TELE'].iloc[0],2) + pow(Blue3['STD_AMP_TELE'].iloc[0],2) + pow(Blue1['STD_SPEAKER_TELE'].iloc[0],2) + pow(Blue2['STD_SPEAKER_TELE'].iloc[0],2) + pow(Blue3['STD_SPEAKER_TELE'].iloc[0],2)))
    blue_match.append([blue_avg_notes, blue_notes_std])
    red_prediction_melody = model.predict(red_match)
    if red_prediction_melody == 1:
        st.write(":red[Red] :green[Melody Achieved]")
    else:   
        st.write(":red[Red] :orange[Melody not Achieved]")
    blue_prediction_melody = model.predict(blue_match)
    if blue_prediction_melody == 1:
        st.write(":blue[Blue] :green[Melody Achieved]")
    else:   
        st.write(":blue[Blue] :orange[Melody not Achieved]")

#@st.cache_data
def melody_rp(red1, red2, red3, blue1, blue2, blue3, team_stats):
    Red1 = team_stats.loc[team_stats['Team Number'] == red1]
    Red2 = team_stats.loc[team_stats['Team Number'] == red2]
    Red3 = team_stats.loc[team_stats['Team Number'] == red3]
    Blue1 = team_stats.loc[team_stats['Team Number'] == blue1]
    Blue2 = team_stats.loc[team_stats['Team Number'] == blue2]
    Blue3 = team_stats.loc[team_stats['Team Number'] == blue3]
    red_spk_avg = Red1['AVG_SPEAKER_TELE'].iloc[0] + Red2['AVG_SPEAKER_TELE'].iloc[0] + Red3['AVG_SPEAKER_TELE'].iloc[0]
    red_amp_avg = Red1['AVG_AMP_TELE'].iloc[0] + Red2['AVG_AMP_TELE'].iloc[0] + Red3['AVG_AMP_TELE'].iloc[0]
    red_avg_notes = red_spk_avg + red_amp_avg
    red_notes_std = math.sqrt(pow(Red1['STD_AMP_TELE'].iloc[0],2) + pow(Red2['STD_AMP_TELE'].iloc[0],2) + pow(Red3['STD_AMP_TELE'].iloc[0],2) + pow(Red1['STD_SPEAKER_TELE'].iloc[0],2) + pow(Red2['STD_SPEAKER_TELE'].iloc[0],2) + pow(Red3['STD_SPEAKER_TELE'].iloc[0],2))
    red_melo = norm.cdf((18-red_avg_notes)/red_notes_std)


    blue_spk_avg = Blue1['AVG_SPEAKER_TELE'].iloc[0] + Blue2['AVG_SPEAKER_TELE'].iloc[0] + Blue2['AVG_SPEAKER_TELE'].iloc[0]
    blue_amp_avg = Blue1['AVG_AMP_TELE'].iloc[0] + Blue2['AVG_AMP_TELE'].iloc[0] + Blue2['AVG_AMP_TELE'].iloc[0]
    blue_avg_notes = blue_spk_avg + blue_amp_avg
    blue_notes_std = math.sqrt(pow(Blue1['STD_AMP_TELE'].iloc[0],2) + pow(Blue2['STD_AMP_TELE'].iloc[0],2) + pow(Blue3['STD_AMP_TELE'].iloc[0],2) + pow(Blue1['STD_SPEAKER_TELE'].iloc[0],2) + pow(Blue2['STD_SPEAKER_TELE'].iloc[0],2) + pow(Blue3['STD_SPEAKER_TELE'].iloc[0],2))
    blue_melo = norm.cdf((18-blue_avg_notes)/blue_notes_std)

    st.write("Red Melody Chance", red_melo)
    st.write("Blue Melod Chance", blue_melo)


#def harmony_rp(red1, red2, red3, blue1, blue2, blue3, team_stats):
def sort_df_row_LTG(row):
    temp = row.copy().values
    attributes = row.columns
    LTG = []
    for _ in range(len(temp)):
        LTG.append(attributes[temp.argmin()])
        temp = np.delete(temp, temp.argmin())
    return LTG    


#def match_table(data):
#   events =
#    sb = statbotics.Statbotics()

    st.write(data.columns)

#def stat_ml_model(p)
#@st.cache_data
def mst_sim_rbt(team, team_stats, weights):
    team_statistics = team_stats.copy().drop(['Win/Total'], axis=1)
    team_number = team_statistics['Team Number']
    # Drop the 'Team Number' column from team_statistics
    #team_statistics.drop(columns=['Team Number', 'Win/Total', 'AVG_AMPLIF_TELE', 'STD_AMPLIF_TELE'], inplace=True)
    # Scale the remaining columns using MinMaxScaler
    scaler = MinMaxScaler()
    #Apply Weights      

    team_standardized = pd.DataFrame(scaler.fit_transform(team_statistics), columns=team_statistics.columns).values
    

    broadcast_weights = np.tile(weights, (team_standardized.shape[0],1))
    weighed_array = team_standardized * broadcast_weights

    team_standardized = pd.DataFrame(weighed_array, columns=team_statistics.columns)
    #teams_weighted = 

    """
    Feeder:

    AVG_AMP_AUTO: 0.01
    AVG_SPEAKER_AUTO: 0.02
    AVG_AMP_TELE: 0.05
    AVG_SPEAKER_TELE: 0.1
    STD_AMP_AUTO: 0.01
    STD_SPEAKER_AUTO: 0.03
    STD_AMP_TELE: 0.03
    STD_SPEAKER_TELE: 0.05
    Predicted Score: 0.05
    Score Variability: 0.05
    AVG_FEED_TELE: 0.3
    AVG_FEED_STD: 0.3
    """

    # Add back the 'Team Number' column
    team_standardized['Team Number'] = team_number
    # Reorder columns to original order
    team_standardized = team_standardized[['Team Number'] + [col for col in team_standardized.columns if col != 'Team Number']]
    
    team_vect = team_standardized.loc[team_standardized['Team Number'] == team].drop(columns=['Team Number'])
    features = team_standardized.loc[team_standardized['Team Number'] == team].drop(columns=['Team Number']).columns
    rankings = []
    strength_rank = []
    for i in range(len(team_standardized)):
        comp_team = team_standardized.iloc[i]['Team Number']
        comp_team_vec = team_standardized.iloc[i].drop(labels='Team Number')
        attribute_similarity = (team_vect - comp_team_vec).abs()

        #st.write(attribute_similarity.min())
        #st.write("X", attribute_similarity)
        #st.write("Y", comp_team_vec.values)
        #st.write("Z", team_vect.values)
        #low = sort_df_row_LTG(attribute_similarity)
        distance = euclidean_distances(team_vect.values, [comp_team_vec.values])
        strength_strat = np.linalg.norm(comp_team_vec.values)
        #st.write(strength_strat)
        strengths = [comp_team, strength_strat, comp_team_vec]
        strength_rank.append(strengths)
        team_and_distance = [comp_team, distance, attribute_similarity]
        rankings.append(team_and_distance)  
    rankings = sorted(rankings, key=lambda x: x[1])[1:10]
    strength_rank = sorted(strength_rank, key=lambda x:x[1], reverse=True)[1:10]
    with st.expander("Best Replacement Robots"):
        list(map(lambda x: st.write(f':red[{str(x[0])}]', x[1], "Difference Row: ",x[2]), rankings))
    with st.expander("Strongest Robots"):
        list(map(lambda x: st.write(f':red[{str(x[0])}]', x[1], "Strength Row: ",x[2]), strength_rank))



    ############ 0 1 complete ###########
    #Euclidean Distances

    
    #Considerations:
    #Hang avg, trap avg, std.devs for all, successful multiple hangs :-(2,3) ;;; 
    #FE
    #STD
    #split
    #train
    #test
    #team_specific = team_stats.loc[team_stats['Team Number'] == team]
    #features = 

@st.cache_data
def plot(team_stats):
    avg_speaker = []
    avg_amp = []
    std_speaker = []
    std_amp = []
    number_of_wins = []
    teams = []
    for i in range(len(team_stats)):
        avg_speaker.append(team_stats.at[i, 'AVG_SPEAKER_AUTO'] + team_stats.at[i, 'AVG_SPEAKER_TELE'])
        avg_amp.append(team_stats.at[i, 'AVG_AMP_AUTO'] + team_stats.at[i, 'AVG_AMP_TELE'])
        std_speaker.append(sqrt(pow(team_stats.at[i, 'STD_SPEAKER_AUTO'],2) + pow(team_stats.at[i, 'STD_SPEAKER_TELE'],2)))
        std_speaker.append(sqrt(pow(team_stats.at[i, 'STD_AMP_AUTO'],2) + pow(team_stats.at[i, 'STD_AMP_TELE'],2)))
        number_of_wins.append(team_stats.at[i,'Win/Total'])
        teams.append(str(team_stats.at[i, 'Team Number']))

    team_data = {
        'Team Number': teams,
        'AVG_SPEAKER': avg_speaker,
        'AVG_AMP':avg_amp,
        'Win/Total': number_of_wins
    }
    df = pd.DataFrame(team_data)
    teams = team_stats['Team Number'].to_list()
    ###################################################################################################################
    # Bubble Plot
    fig = px.scatter(df, x="AVG_AMP", y="AVG_SPEAKER",
	         size="Win/Total", color="Team Number",
                 hover_name="Team Number", log_x=True, title="Win/Total vs Avg Amp and Avg")
    #fig.update_traces(marker_size=10)

    return fig

    ###################################################################################################################
    # Analyze the best alliance setup


 
def auto_paths(Data, teams):
    pickup_points = {
        "auto_note_3_pickup": (103, 250.1),
        "auto_note_2_pickup": (103, 198.4),
        "auto_note_1_pickup": (103, 145.8),
        "auto_note_4_pickup": (295.6, 266.8),
        "auto_note_5_pickup": (295.6, 206.7),
        "auto_note_6_pickup": (295.6, 147),
        "auto_note_7_pickup": (295.6, 86.2),
        "auto_note_8_pickup": (295.6, 25.7),
        "auto_note_11_pickup": (488.6, 250.1),
        "auto_note_10_pickup": (488.6, 198.4),
        "auto_note_9_pickup": (488.6, 145.8)
    }

    scale_factor_x = 1
    scale_factor_y = 1

    fig = go.Figure()


    with open("Auto Path References (1).png", "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('ascii')

    fig.add_layout_image(
        go.layout.Image(
            source=f"data:image/png;base64,{img_base64}",
            xref="x",
            yref="y",
            x=0,
            y=3,
            sizex=600,
            sizey=260,
            sizing="stretch",
            layer="below")
    )

    all_route_x = []  # List to store x coordinates of all points
    all_route_y = []  # List to store y coordinates of all points
    for r in range(len(teams)): 

        team_locs = Data.loc[Data['team_#'] == teams[r]]
        for i in range(len(team_locs)):

            locs = team_locs.iloc[i][['auto_note_1_pickup','auto_note_2_pickup', 'auto_note_3_pickup', 'auto_note_4_pickup', 'auto_note_5_pickup', 'auto_note_6_pickup', 'auto_note_7_pickup', 'auto_note_8_pickup', 'auto_note_9_pickup', 'auto_note_10_pickup', 'auto_note_11_pickup']]
            note_locs = ['auto_note_1_pickup','auto_note_2_pickup', 'auto_note_3_pickup', 'auto_note_4_pickup', 'auto_note_5_pickup', 'auto_note_6_pickup', 'auto_note_7_pickup', 'auto_note_8_pickup', 'auto_note_9_pickup', 'auto_note_10_pickup', 'auto_note_11_pickup']
            locations = []

            for z in range(len(note_locs)):
                if type(locs[note_locs[z]]) == np.float64:
                    if math.isnan(float(locs[note_locs[z]])) == False:
                        locations.append([note_locs[z], locs[note_locs[z]]])                      
                
            index_to_sort_by = 1

            locations = sorted(locations, key=lambda x: x[index_to_sort_by])

            route_x = []
            route_y = []


            #min = 10000000000000
            #for r in range(len(locations)):
            start_pos = team_locs.iloc[i]['start_pos']
            if team_locs.iloc[i]['alliance_color'] == 'blue':
                if start_pos == 'Center':
                    route_x.append(39.4)
                    route_y.append(198)
                elif start_pos == 'Source':
                    route_x.append(39.4)
                    route_y.append(144.2)
                elif start_pos == 'Amp':
                    route_x.append(39.4)
                    route_y.append(252.1)
            elif team_locs.iloc[i]['alliance_color'] == 'red':
                if start_pos == 'Center':
                    route_x.append(552.6)
                    route_y.append(198)
                elif start_pos == 'Source':
                    route_x.append(552.6)
                    route_y.append(144.2)
                elif start_pos == 'Amp':
                    route_x.append(552.6)
                    route_y.append(252.1)
                #if start_pos == 
            
            
            for node in range(len(locations)):
                route_x.append(pickup_points[locations[node][0]][0])
                route_y.append(pickup_points[locations[node][0]][1])
            all_route_x.append(route_x)
            all_route_y.append(route_y)
            
            str_team = np.vectorize(str)(teams)

            for x in range(len(all_route_y)):
                routes_xs = pd.Series(all_route_x[x])
                routes_ys = pd.Series(all_route_y[x])
                fig.add_scatter(
                    x=all_route_x[i],
                    y=all_route_y[i],
                    mode='lines',
                    name=f"Team {teams[r]}"

                )

                fig.add_scatter(
                    x=all_route_x[x],
                    y=all_route_y[x]
                )
    fig.update_layout(
    xaxis_title='X-Position',
    yaxis_title='Y-Position',
    title=f'Auto Paths for {teams}',
    showlegend=True

    )
    fig.add_layout_image
    return fig
            

    # Find if a robot is better at x

 
def max_amp_spk(clean_data, team):
    maxes = {}

    for tm in team:
        team_df = clean_data.loc[clean_data['team_#'] == tm]
        amplif_spks = []

        for index, row in team_df.iterrows():  # Use iterrows() to iterate over rows
            row.fillna(0, inplace=True)
            amplifieds = row['AMPLIFIED'].split('|') if '|' in str(row['AMPLIFIED']) else [row['AMPLIFIED']]
            stop_amps = row['amplified_closed'].split('|') if '|' in str(row['amplified_closed']) else [row['amplified_closed']]
            spks = row['speaker_scored_amped'].split('|') if '|' in str(row['speaker_scored_amped']) else [row['speaker_scored_amped']]
            st.write(amplifieds)
            st.write(stop_amps)
            for i in range(len(amplifieds)):
                if i <= len(stop_amps) - 1:
                    amplif_spks.append(len([x for x in spks if amplifieds[i] < x < stop_amps[i]]))
        maxes.update({f"{tm}" : max(amplif_spks)})

    st.write(maxes)


    

 
def node_graph_picklist(team_stats):
    team_to = {

    }
    for i in range(len(team_stats)):

        data = team_stats.loc[team_stats['Team Number'] != team_stats.at[i, 'Team Number']]
        opos_team = []
        teams = []
        for j in range(len(data['Team Number'].unique())):
            opos_team.append(str(data['Team Number'].unique()[j]))

        teams.append(opos_team)
    return teams



  
def tele_amp(data, teams):
    fig = go.Figure()
    for i in range(len(teams)):
        team_stats = data.loc[data['team_#'] == teams[i]]
        matches = list(range(1, len(team_stats)))
        fig.add_scatter(
            x=matches,
            y=team_stats['tele_amp_scored'],
            name=f"Team {teams[i]}"
        )
    fig.update_layout(
    xaxis_title='Matches',
    yaxis_title='Tele-Amp Score',
    title='Tele-Amp Scores over matches for Teams',
    showlegend=True
    )
    return fig

 
def tele_speaker(data, teams):
    fig = go.Figure()
    for i in range(len(teams)):
        team_stats = data.loc[data['team_#'] == teams[i]]
        matches = list(range(1, len(team_stats)))
        fig.add_scatter(
            x=matches,
            y=team_stats['tele_spk_scored'],
            name=f"Team {teams[i]}"
        )
    fig.update_layout(
    xaxis_title='Matches',
    yaxis_title='Tele-Speaker Score',
    title='Tele-Speaker Scores over matches for Teams',
    showlegend=True
    )
    return fig


 
def auto_amp(data, teams):
    fig = go.Figure()
    for i in range(len(teams)):
        team_stats = data.loc[data['team_#'] == teams[i]]
        matches = list(range(1, len(team_stats)))
        fig.add_scatter(
            x=matches,
            y=team_stats['auto_amp_scored'],
            name=f"Team {teams[i]}"
        )
    fig.update_layout(
    xaxis_title='Matches',
    yaxis_title='Auto-Amp Score',
    title='Auto-Amp Scores over matches for Teams',
    showlegend=True
    )
    return fig

 
def auto_speaker(data, teams):
    fig = go.Figure()
    for i in range(len(teams)):
        team_stats = data.loc[data['team_#'] == teams[i]]
        matches = list(range(1, len(team_stats)))
        fig.add_scatter(
            x=matches,
            y=team_stats['auto_spk_scored'],
            name=f"Team {teams[i]}"
        )
    fig.update_layout(
    xaxis_title='Matches',
    yaxis_title='Auto-Speaker Score',
    title='Auto-Speaker Scores over matches for Teams',
    showlegend=True
    )
    return fig
'''def tele_radar(data_team):
    fig = go.Figure()
    for i in rag'''

        




    
#Test Data: [[A1,Null],[A2, 5], [A3, 10], [A4, Null], [A5, 2.7]]
