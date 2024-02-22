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
#from tensorflow import keras

#'Scouting\Team_Analysis\default (1).csv'
def get_clean_data(csv):
    default = pd.read_csv(csv)
    attributes = ['tele_spk_scored', 'tele_amp_scored',]
    for i in range(len(default)):
        for j in range(len(attributes)):
            if pd.isna(default.loc[i, attributes[j]]) or len(default.at[i, attributes[j]]) == 0:
                default.at[i,attributes[j]]=0
            else:
                default.loc[i, attributes[j]] = len(default.at[i, attributes[j]].split('|'))
    times = ['AMPLIFIED', 'amplified_closed']
    for r in range(len(default)):
        for c in range(len(times)):
            if pd.isna(default.loc[r, times[c]]) or len(default.at[r, times[c]]) == 0:
                default.at[r, times[c]] == [-1]
            else:
                print("Inside else on line 32")
                print("r;", r)
                print("c;", c)
                print("val;", default.at[r, times[c]])
                print(default.at[r, "AMPLIFIED"])
                print(default.at[r, 'amplified_closed'])
                print(times[c])
                try:
                    default.loc[r, times[c]] = default.at[r, times[c]].split('|')
                except:
                    default.loc[r, times[c]] = [-1]
    return default
#team_stats = pd.DataFrame(columns=['Team-Number', 'AVG_AMP_AUTO', 'AVG_SPEAKER_AUTO', 'AVG_AMP_TELE', 'AVG_SPEAKER_TELE', 'STD_AMP_AUTO', 'STD_SPEAKER_AUTO', 'STD_AMP_TELE', 'STD_SPEAKER_TELE' ])
def team_desc(Data):
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
    team_stats.insert(10, 'Score Variablity', 0)
    team_stats.insert(11, 'Wins', 0)
    for i in range(len(team_stats)):
        team = team_stats.at[i, 'Team Number']
        Team_DF = Data.loc[Data['team_#'] == team]
        team_stats.loc[i, 'AVG_AMP_AUTO'] = Team_DF['auto_amp_scored'].describe()[1]
        team_stats.loc[i, 'AVG_SPEAKER_AUTO'] = Team_DF['auto_spk_scored'].describe()[1]
        team_stats.loc[i, 'AVG_AMP_TELE'] = Team_DF['tele_amp_scored'].describe()[1]
        team_stats.loc[i, 'AVG_SPEAKER_TELE'] = Team_DF['tele_spk_scored'].describe()[1]

        team_stats.loc[i, 'STD_AMP_AUTO'] = Team_DF['auto_amp_scored'].describe()[2]
        team_stats.loc[i, 'STD_SPEAKER_AUTO'] = Team_DF['auto_spk_scored'].describe()[2]
        team_stats.loc[i, 'STD_AMP_TELE'] = Team_DF['tele_amp_scored'].describe()[2]
        team_stats.loc[i, 'STD_SPEAKER_TELE'] = Team_DF['tele_spk_scored'].describe()[2]
    for j in range(len(team_stats)):
        #if team_stats.at[j, 'Team Number'] == 1234:    
        team_stats.loc[j,'Predicted Score'] = team_stats.at[j, 'AVG_AMP_AUTO']*2 + team_stats.at[j, 'AVG_SPEAKER_AUTO']*5 + team_stats.at[j, 'AVG_AMP_TELE']*1 + team_stats.at[j, 'AVG_SPEAKER_TELE']*2
        team_stats.loc[j, 'Score Variability'] = math.sqrt(pow(team_stats.at[j,'STD_AMP_AUTO']*2, 2) + pow(team_stats.at[j,'STD_SPEAKER_AUTO']*5, 2) + pow(team_stats.at[j,'STD_AMP_TELE']*1, 2) + pow(team_stats.at[j,'STD_SPEAKER_TELE'] * 2, 2) )
        if pd.isna(team_stats.loc[j, 'Score Variablity']):
            team_stats.loc[j, 'Score Variablity'] == 0
    team_stats.at[0, 'Wins'] = 5
    team_stats.at[1, 'Wins'] = 10
    team_stats.at[2, 'Wins'] = 15
    team_stats.at[3, 'Wins'] = 20
    return team_stats

def match_prediction(team_stats):
#print("Type your 6 teams in order from 0-9")
    cont = input("Want Matche Predictions(yes/no)?")
#st.title('Scouting Alliance Win Predictions')
    while cont =='yes':
        print("Type your 6 teams in order from 0-9")
        Red1 = int(input("Input Red1: "))    
        Red2 = int(input("Input Red2: "))    
        Red3 = int(input("Input Red3: "))    
        Blue1 = int(input("Input Blue1: "))    
        Blue2 = int(input("Input Blue2: "))    
        Blue3 = int(input("Input Blue3: "))
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

        print('Blue Chance of Winning')
        print(Blue_Win * 100)
        print('Red Chance of Winning')
        print((1-Blue_Win) * 100)
        #Scouting\Team_Analysis\PointCalculator.py


        cont = input("Want more match data(yes/no)?")


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
        number_of_wins.append(team_stats.at[i,'Wins'])
        teams.append(team_stats.at[i, 'Team Number'])

    team_data = {
        'Team Number': teams,
        'AVG_SPEAKER': avg_speaker,
        'AVG_AMP':avg_amp,
        'Wins': number_of_wins
    }
    
    df = pd.DataFrame(team_data)
    teams = team_stats['Team Number'].to_list()
    sizes = np.array(number_of_wins) * 50
    ###################################################################################################################
    # Bubble Plot
    fig = px.scatter(df, x='AVG_SPEAKER', y='AVG_AMP', size='Wins', hover_data=['Team Number'])
    #plt.scatter(std_speaker, std_amp, s=number_of_wins*10, alpha=0.5, c='black', edgecolors='black')
    fig.update_layout(
    title='Bubble Plot of Wins vs. AVG_SPEAKER and AVG_AMP',
    xaxis_title='AVG_SPEAKER',
    yaxis_title='AVG_AMP'
    )
    fig.show()

def auto_paths(Data):
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

    more = 'True'
    img = mpimg.imread('FRC_Crescendo_Field.png')
    image_width = img.shape[1]  # Width of the image
    image_height = img.shape[0]

    scale_factor_x = 1
    scale_factor_y = 1

    while more == 'True':
        teams = input("What team's auto path would you like(Only Numbers seperated by spaces): ").split()
        teams = [eval(i) for i in teams]
        fig, ax = plt.subplots()

        ax.imshow(img, extent=[0, image_width*scale_factor_x, 0, image_height*scale_factor_y])
        all_route_x = []  # List to store x coordinates of all points
        all_route_y = []  # List to store y coordinates of all points
        for r in range(len(teams)): 

            team_locs = Data.loc[Data['team_#'] == teams[r]]
            for i in range(len(team_locs)):

                locs = team_locs.iloc[i][['auto_note_1_pickup','auto_note_2_pickup', 'auto_note_3_pickup', 'auto_note_4_pickup', 'auto_note_5_pickup', 'auto_note_6_pickup', 'auto_note_7_pickup', 'auto_note_8_pickup', 'auto_note_9_pickup', 'auto_note_10_pickup', 'auto_note_11_pickup']]
                note_locs = ['auto_note_1_pickup','auto_note_2_pickup', 'auto_note_3_pickup', 'auto_note_4_pickup', 'auto_note_5_pickup', 'auto_note_6_pickup', 'auto_note_7_pickup', 'auto_note_8_pickup', 'auto_note_9_pickup', 'auto_note_10_pickup', 'auto_note_11_pickup']
                locations = []

                for z in range(len(note_locs)):
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
                all_route_x.extend(route_x)
                all_route_y.extend(route_y)
            try:
                # Calculate KDE for all points
                xy = np.vstack([all_route_x, all_route_y])
                z = gaussian_kde(xy)(xy)

                # Create heatmap for lines
                for i in range(len(all_route_x) - 1):
                    x_values = np.linspace(all_route_x[i], all_route_x[i + 1], 5)  # Interpolating x coordinates
                    y_values = np.linspace(all_route_y[i], all_route_y[i + 1], 5)  # Interpolating y coordinates
                    for j in range(len(x_values) - 1):
                        density = gaussian_kde(xy)([[x_values[j]], [y_values[j]]])[0]
                        line_color = plt.cm.hot(density)
                        ax.plot([x_values[j], x_values[j + 1]], [y_values[j], y_values[j + 1]], color=line_color, linewidth=4, alpha= 0.5)

                # Create heatmap for scattered points
                ax.scatter(all_route_x, all_route_y, color='blue', alpha=0.5, s=150)  # Scatter plot for scattered points
            except:
                ax.scatter(all_route_x, all_route_y, color='blue', alpha=0.5, s=150)  # Scatter plot for scattered points

            #ax.plot(route_x, route_y, alpha=0.5, linewidth=3.5)
            #ax.scatter(route_x, route_y, color='red', s=100, alpha=0.5)  # Adjust 's' to change the size of the circles


            #ax.set_aspect('equal')
            ax.set_xlim(0, image_width*scale_factor_x)
            ax.set_ylim(0, image_height*scale_factor_y)
            plt.title(str(teams[r]) + " Autopaths")
        plt.show()
        more = (input("Want more team auto_paths?(True, False): "))
            


    # Find if a robot is better at 


def tables(Data, team_stats):

    print(team_stats)
    # Define cellText and column headers
    Data = Data.round(2)
    cellText = Data.values
    colLabels = Data.columns

    # Create a matplotlib figure
    fig, ax = plt.subplots()

    # Create the table and add it to the figure
    table = ax.table(cellText=cellText, colLabels=colLabels, loc='center')
    # Adjust table appearance (optional)
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.5)
    # Create a Tele
    #table2 = ax.table(cellText=cellText, colLabels = )
    # Show the plot
    plt.show()

def amp(clean_data):
    #def handle():


    team = int(input("Team: "))
    clean_data = clean_data.loc[clean_data['team_#'] == team]
    amp_start = clean_data['AMPLIFIED']
    amp_end = clean_data['amplified_closed']
    print(amp_start)
    print(amp_end)
    start_end = []    
    for i in range(len(amp_start)):
        try:
            start_end.append(amp_start.at[i, 'AMPLIFIED'])
            start_end.append(amp_end.at[i, 'amplified_close'])
        except ValueError:
            #amp_start.at[i, 'AMPLIFIED'],   = handle()
            start_end.append(amp_start.at[i, 'AMPLIFIED'])
            start_end.append(amp_end.at[i, 'amplified_close'])
    print(start_end)

def simulations(team_stats):
    num_times = 10000
    teams = int(input("Teams in match(Red => Blue): ")).split()
    master_simulations = []
    for i in range(len(teams)):
        samples = np.random.sample(team_stats['Predicted'], team_stats['Std.D'], num_times)

 
        




        





#Test Data: [[A1,Null],[A2, 5], [A3, 10], [A4, Null], [A5, 2.7]]