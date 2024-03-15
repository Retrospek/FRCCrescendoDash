import pandas as pd
import numpy as np
import math
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

#'Scouting\Team_Analysis\default (1).csv'
@st.cache_data
def get_clean_data(data):
    
    default = data

    attributes = ['tele_spk_scored', 'tele_amp_scored', 'speaker_scored_amped']
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

    times = ['AMPLIFIED', 'amplified_closed']
    for r in range(len(default)):
        for c in range(len(times)):
            if pd.isna(default.loc[r, times[c]]):
                default.at[r, times[c]] == [-1]
            else:
                try:
                    default.loc[r, times[c]] = default.at[r, times[c]].split('|')
                except:
                    default.loc[r, times[c]] = [-1]
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
    team_stats.insert(9, 'AVG_AMPLIF_TELE', 0)
    team_stats.insert(10, 'STD_AMPLIF_TELE', 0)
    team_stats.insert(11, 'Predicted Score', 0)
    team_stats.insert(12, 'Score Variability', 0)
    team_stats.insert(13, 'Win/Total', 0)
    for i in range(len(team_stats)):
        team = team_stats.at[i, 'Team Number']
        Team_DF = Data.loc[Data['team_#'] == team]
        team_wins = len(Team_DF.loc[Team_DF['win'] == 'Yes'])
        team_total = len(Team_DF)
        team_stats.loc[i, 'AVG_AMP_AUTO'] = Team_DF['auto_amp_scored'].describe()[1]
        team_stats.loc[i, 'AVG_SPEAKER_AUTO'] = Team_DF['auto_spk_scored'].describe()[1]
        team_stats.loc[i, 'AVG_AMP_TELE'] = Team_DF['tele_amp_scored'].describe()[1]
        team_stats.loc[i, 'AVG_SPEAKER_TELE'] = Team_DF['tele_spk_scored'].describe()[1]
        team_stats.loc[i, 'AVG_AMPLIF_TELE'] = Team_DF['speaker_scored_amped'].describe()[1]

        team_stats.loc[i, 'STD_AMP_AUTO'] = Team_DF['auto_amp_scored'].describe()[2]
        team_stats.loc[i, 'STD_SPEAKER_AUTO'] = Team_DF['auto_spk_scored'].describe()[2]
        team_stats.loc[i, 'STD_AMP_TELE'] = Team_DF['tele_amp_scored'].describe()[2]
        team_stats.loc[i, 'STD_SPEAKER_TELE'] = Team_DF['tele_spk_scored'].describe()[2]
        team_stats.loc[i, 'STD_AMPLIF_TELE'] = Team_DF['speaker_scored_amped'].describe()[2]


        team_stats.loc[i, 'Win/Total'] = team_wins/team_total
    team_stats['AVG_AMP_AUTO'].fillna(0, inplace=True)
    team_stats['AVG_SPEAKER_AUTO'].fillna(0, inplace=True)
    team_stats['AVG_AMP_TELE'].fillna(0, inplace=True)
    team_stats['AVG_SPEAKER_TELE'].fillna(0, inplace=True)
    team_stats['AVG_AMPLIF_TELE'].fillna(0, inplace=True)
    team_stats['STD_AMP_AUTO'].fillna(0, inplace=True)
    team_stats['STD_SPEAKER_AUTO'].fillna(0, inplace=True)
    team_stats['STD_AMP_TELE'].fillna(0, inplace=True)
    team_stats['STD_SPEAKER_TELE'].fillna(0, inplace=True)
    team_stats['STD_AMPLIF_TELE'].fillna(0, inplace=True)
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

def ml_clean(team_stats):
    training = pd.DataFrame(team_stats['Team Number'], columns=['Team Number'])
    training['AUTO_AVG'] = team_stats[['AVG_SPEAKER_AUTO', 'AVG_AMP_AUTO']].sum(axis=1)
    training['AVG_TELE_AMP'] = team_stats['AVG_AMP_TELE']
    training['AVG_TELE_SPEAKER'] = team_stats['AVG_SPEAKER_TELE']
    training['VARIABILITY'] = team_stats['Score Variability']
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(training.drop('Team Number', axis=1))
    normalized_df = pd.DataFrame(normalized_data, columns=['AUTO_AVG', 'AVG_TELE_SPEAKER', 'AVG_TELE_AMP', 'VARIABILITY'])
    
    full_data = pd.concat([training['Team Number'], normalized_df], axis=1) 

    return full_data
    #st.dataframe(X)

@st.cache_data
def get_matches_cleaned():
    file_path = "data.txt"

    total_df = pd.DataFrame(columns=['Red1', 'Red2', 'Red3', 'Blue1', 'Blue2', 'Blue3', 'RedScore', 'BlueScore'])
    with open(file_path, 'r') as file:
        # Read the file line by line
        line_count = 0
        for line in file:
            line_count += 1

            if line_count % 2 == 0:
                # Split the line into individual values
                values = line.strip().split('\t')
                values = [float(val) if val.replace('.', '', 1).isdigit() else None for val in values]
                # Append the new DataFrame to the total_df
                total_df.loc[len(total_df.index)] = values 


    return total_df

@st.cache_data
def ml_data(matches, stand_teams):
    matches['Winner'] = 'Red'  # Assume Red alliance won initially
    matches.loc[matches['BlueScore'] > matches['RedScore'], 'Winner'] = 'Blue'  # Update to 'Blue' if Blue alliance won
    aggreg = matches.drop(['RedScore', 'BlueScore'], axis=1)
    #aggreg['Team Number'] = aggreg['Team Number'].astype('float64')
    ##### Created the match outcomes
    ##### Now to create training dataset
    x_train = []
    y_train = []
    attributes = ['AUTO_AVG', 'AVG_TELE_SPEAKER', 'AVG_TELE_AMP', 'VARIABILITY']
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
            if attribute == 'VARIABILITY':
                RedAlliance.append(math.sqrt(pow(Red1[attribute].iloc[0], 2)+pow(Red2[attribute].iloc[0], 2)+pow(Red3[attribute].iloc[0], 2)))
                BlueAlliance.append(math.sqrt(pow(Blue1[attribute].iloc[0], 2)+pow(Blue2[attribute].iloc[0], 2)+pow(Blue3[attribute].iloc[0], 2)))

            else:
                RedAlliance.append(Red1[attribute].iloc[0] + Red2[attribute].iloc[0] + Red3[attribute].iloc[0])
                BlueAlliance.append(Blue1[attribute].iloc[0] + Blue2[attribute].iloc[0] + Blue3[attribute].iloc[0])
                
    
        difference = [x - y for x, y in zip(RedAlliance, BlueAlliance)]
        result = []

        if aggreg.at[i, 'Winner'] == 'Red':
             result.append(0)
        else:
            result.append(1)
        x_train.append(difference)
        y_train.append(result)

    return x_train, y_train

@st.cache_data
def ml_model(X_TRAIN, Y_TRAIN):
    X_train, X_test, y_train, y_test = train_test_split(X_TRAIN, Y_TRAIN, test_size=0.2, random_state=42)
    gb_clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2, random_state=42)
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(gb_clf, 'rf_model.joblib')

    loaded_model = joblib.load('rf_model.joblib')

    y_pred_two = loaded_model.predict(X_TRAIN)
    accuracy = accuracy_score(Y_TRAIN, y_pred_two)
    st.write("Waco Tournament Accuracy:", accuracy)

def use_model(Red_teams, Blue_teams, stand_teams):
    loaded_model = joblib.load('rf_model.joblib')
    Red1 = stand_teams.loc[stand_teams['Team Number'] == Red_teams[0]]
    Red2 = stand_teams.loc[stand_teams['Team Number'] == Red_teams[1]]
    Red3 = stand_teams.loc[stand_teams['Team Number'] == Red_teams[2]]
    Blue1 = stand_teams.loc[stand_teams['Team Number'] == Blue_teams[0]]
    Blue2 = stand_teams.loc[stand_teams['Team Number'] == Blue_teams[1]]
    Blue3 = stand_teams.loc[stand_teams['Team Number'] == Blue_teams[2]]
    attributes = ['AUTO_AVG', 'AVG_TELE_SPEAKER', 'AVG_TELE_AMP', 'VARIABILITY']
    RedAlliance = []
    BlueAlliance = []
    for j in range(len(attributes)):
        attribute = attributes[j]
        if attribute == 'VARIABILITY':
            RedAlliance.append(math.sqrt(pow(Red1[attribute].iloc[0], 2)+pow(Red2[attribute].iloc[0], 2)+pow(Red3[attribute].iloc[0], 2)))
            BlueAlliance.append(math.sqrt(pow(Blue1[attribute].iloc[0], 2)+pow(Blue2[attribute].iloc[0], 2)+pow(Blue3[attribute].iloc[0], 2)))

        else:
            RedAlliance.append(Red1[attribute].iloc[0] + Red2[attribute].iloc[0] + Red3[attribute].iloc[0])
            BlueAlliance.append(Blue1[attribute].iloc[0] + Blue2[attribute].iloc[0] + Blue3[attribute].iloc[0])
            
    difference = [x - y for x, y in zip(RedAlliance, BlueAlliance)]
    x_test = []
    x_test.append(difference)
    match_pred = loaded_model.predict(x_test)
    if match_pred == 0:
        st.write('Prediction: RED') 
    else:
        st.write('Prediction: BLUE')



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
