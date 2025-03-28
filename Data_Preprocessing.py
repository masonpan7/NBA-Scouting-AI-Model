import numpy as np 
import pandas as pd
import os

# Check if the file exists
file_path = 'Seasons_Stats.csv'
if os.path.exists(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
else:
    print(f"File {file_path} does not exist.")

file_path = 'player_data.csv'
if os.path.exists(file_path):
    # Load the dataset
    df_player_data = pd.read_csv(file_path)
else:
    print(f"File {file_path} does not exist.")

# Convert 'Year' column to numeric, handling potential errors
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Identify players who played in 2012 or later
players_2012_or_later = df[df['Year'] >= 2012]['Player'].unique()

# Filter the dataframe to include all data for those players
df = df[df['Player'].isin(players_2012_or_later)]

# Display the first few rows after filtering
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df.reset_index(drop=True, inplace=True)
df = df.sort_values(by=['Player', 'Year'], ascending=[True, True])
df = df.reset_index(drop=True)

# Add BPM and VORP to the features list
features = [
    "Year", "Player", "Pos", "Age", "G", "MP", "PER", "TS%", "WS", "WS/48", "3PAr", "FTr", "USG%",
    "FGA", "FG%", "TRB", "AST", "STL", "BLK", "TOV", "PTS", "BPM", "VORP"
]
df = df[features]

# Group by Player and Year, summing the statistics
df_grouped = df.groupby(['Player', 'Year']).agg({
    'Pos': 'first',  # Assuming position doesn't change
    'Age': 'first',  # Assuming age doesn't change within a year
    'G': 'sum',
    'MP': 'sum',
    'PER': 'mean',  # Averaging PER
    'TS%': 'mean',  # Averaging TS%
    'WS': 'sum',
    'WS/48': 'mean',  # Averaging WS/48
    'FGA': 'sum',
    'FG%': 'mean',  # Averaging FG%
    'TRB': 'sum',
    'AST': 'sum',
    'STL': 'sum',
    'BLK': 'sum',
    'TOV': 'sum',
    'PTS': 'sum',
    'USG%': 'mean',  # Averaging USG%
    '3PAr': 'mean',  # Averaging 3PAr
    'BPM': 'mean',  # Averaging BPM
    'VORP': 'mean'  # Averaging VORP
}).reset_index()

# Calculate per-game statistics after grouping
df_grouped['MP/G'] = (df_grouped['MP'] / df_grouped['G']).round(2)
df_grouped['TRB/G'] = (df_grouped['TRB'] / df_grouped['G']).round(2)
df_grouped['AST/G'] = (df_grouped['AST'] / df_grouped['G']).round(2)
df_grouped['STL/G'] = (df_grouped['STL'] / df_grouped['G']).round(2)
df_grouped['BLK/G'] = (df_grouped['BLK'] / df_grouped['G']).round(2)
df_grouped['TOV/G'] = (df_grouped['TOV'] / df_grouped['G']).round(2)
df_grouped['PTS/G'] = (df_grouped['PTS'] / df_grouped['G']).round(2)

# Round BPM and VORP to two decimal places
df_grouped['BPM'] = df_grouped['BPM'].round(2)
df_grouped['VORP'] = df_grouped['VORP'].round(2)

# Round per-game statistics to the second decimal place and remove trailing zeros
df_grouped['MP/G'] = df_grouped['MP/G'].round(2).apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
df_grouped['TRB/G'] = df_grouped['TRB/G'].round(2).apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
df_grouped['AST/G'] = df_grouped['AST/G'].round(2).apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
df_grouped['STL/G'] = df_grouped['STL/G'].round(2).apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
df_grouped['BLK/G'] = df_grouped['BLK/G'].round(2).apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
df_grouped['TOV/G'] = df_grouped['TOV/G'].round(2).apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))
df_grouped['PTS/G'] = df_grouped['PTS/G'].round(2).apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))

# Sort the data by player name, starting year, and ending year
df_player_data = df_player_data[['name', 'year_start', 'year_end']]
df_player_data = df_player_data[df_player_data['year_end'] >= 2012]
df_player_data.sort_values(by=['name'], ascending=[True], inplace=True) 
df_player_data.reset_index(drop=True, inplace=True)

df_grouped['is_first_year'] = 0
df_grouped['PTS/G_change'] = 0.0
df_grouped['PER_change'] = 0.0
df_grouped['WS_change'] = 0.0
df_grouped['TRB_change'] = 0.0
df_grouped['BLK_change'] = 0.0
df_grouped['AST_change'] = 0.0
df_grouped['STL_change'] = 0.0
df_grouped['TS%_change'] = 0.0
df_grouped['USG%_change'] = 0.0
df_grouped['MP/G_change'] = 0.0
df_grouped['TOV/G_change'] = 0.0
df_grouped['FG%_change'] = 0.0
df_grouped['3PA/G_change'] = 0.0
df_grouped['WS/48_change'] = 0.0

for i in range(len(df_grouped)):
    player = df_grouped.loc[i, 'Player']
    year = df_grouped.loc[i, 'Year']
    
    # Identify if it's the player's first year
    if not df_player_data[(df_player_data['name'] == player) & (df_player_data['year_start'] == year)].empty:
        df_grouped.loc[i, 'is_first_year'] = 1
    
    # Calculate changes in various statistics
    previous_year_data = df_grouped[(df_grouped['Player'] == player) & (df_grouped['Year'] == year - 1)]
    if not previous_year_data.empty:
        df_grouped.loc[i, 'PTS/G_change'] = float(df_grouped.loc[i, 'PTS/G']) - float(previous_year_data['PTS/G'].values[0])
        df_grouped.loc[i, 'PER_change'] = float(df_grouped.loc[i, 'PER']) - float(previous_year_data['PER'].values[0])
        df_grouped.loc[i, 'WS_change'] = float(df_grouped.loc[i, 'WS']) - float(previous_year_data['WS'].values[0])
        df_grouped.loc[i, 'TRB_change'] = float(df_grouped.loc[i, 'TRB/G']) - float(previous_year_data['TRB/G'].values[0])
        df_grouped.loc[i, 'BLK_change'] = float(df_grouped.loc[i, 'BLK/G']) - float(previous_year_data['BLK/G'].values[0])
        df_grouped.loc[i, 'AST_change'] = float(df_grouped.loc[i, 'AST/G']) - float(previous_year_data['AST/G'].values[0])
        df_grouped.loc[i, 'STL_change'] = float(df_grouped.loc[i, 'STL/G']) - float(previous_year_data['STL/G'].values[0])
        df_grouped.loc[i, 'TS%_change'] = float(df_grouped.loc[i, 'TS%']) - float(previous_year_data['TS%'].values[0])
        df_grouped.loc[i, 'USG%_change'] = float(df_grouped.loc[i, 'USG%']) - float(previous_year_data['USG%'].values[0])
        df_grouped.loc[i, 'MP/G_change'] = float(df_grouped.loc[i, 'MP/G']) - float(previous_year_data['MP/G'].values[0])
        df_grouped.loc[i, 'TOV/G_change'] = float(df_grouped.loc[i, 'TOV/G']) - float(previous_year_data['TOV/G'].values[0])
        df_grouped.loc[i, 'FG%_change'] = float(df_grouped.loc[i, 'FG%']) - float(previous_year_data['FG%'].values[0])
        df_grouped.loc[i, '3PA/G_change'] = float(df_grouped.loc[i, '3PAr']) - float(previous_year_data['3PAr'].values[0])
        df_grouped.loc[i, 'WS/48_change'] = float(df_grouped.loc[i, 'WS/48']) - float(previous_year_data['WS/48'].values[0])


df_grouped['PTS/G_change'] = df_grouped['PTS/G_change'].round(2)
df_grouped['PER_change'] = df_grouped['PER_change'].round(2)
df_grouped['WS_change'] = df_grouped['WS_change'].round(2)
df_grouped['TRB_change'] = df_grouped['TRB_change'].round(2)
df_grouped['BLK_change'] = df_grouped['BLK_change'].round(2)
df_grouped['AST_change'] = df_grouped['AST_change'].round(2)
df_grouped['STL_change'] = df_grouped['STL_change'].round(2)
df_grouped['TS%_change'] = df_grouped['TS%_change'].round(2)
df_grouped['USG%_change'] = df_grouped['USG%_change'].round(2)
df_grouped['MP/G_change'] = df_grouped['MP/G_change'].round(2)
df_grouped['TOV/G_change'] = df_grouped['TOV/G_change'].round(2)
df_grouped['FG%_change'] = df_grouped['FG%_change'].round(2)
df_grouped['3PA/G_change'] = df_grouped['3PA/G_change'].round(2)
df_grouped['WS/48_change'] = df_grouped['WS/48_change'].round(2)

# Calculate breakout probability score
def define_breakout(df):
    df['breakout'] = 0

    for i in range(len(df)):
        pos = df.loc[i, 'Pos']
        ppg_change = df.loc[i, 'PTS/G_change']
        per_change = df.loc[i, 'PER_change']
        ws_change = df.loc[i, 'WS_change']
        trb_change = df.loc[i, 'TRB_change']
        blk_change = df.loc[i, 'BLK_change']
        ast_change = df.loc[i, 'AST_change']
        stl_change = df.loc[i, 'STL_change']
        ts_change = df.loc[i, 'TS%_change']
        usg_change = df.loc[i, 'USG%_change']
        mp_change = df.loc[i, 'MP/G_change']
        tov_change = df.loc[i, 'TOV/G_change']
        fg_change = df.loc[i, 'FG%_change']
        threepa_change = df.loc[i, '3PA/G_change']
        ws48_change = df.loc[i, 'WS/48_change']

        criteria_met = 0

        if pos == 'PG':
            if ppg_change >= 4.0:
                criteria_met += 1
            if ast_change >= 1.8:
                criteria_met += 1
            if ts_change >= 2.0:
                criteria_met += 1
            if ws_change >= 2.0:
                criteria_met += 1
            if usg_change >= 3.0:
                criteria_met += 1
            if tov_change <= 0.8:
                criteria_met += 1
            if mp_change >= 5.5:
                criteria_met += 1
            if stl_change >= 0.5:
                criteria_met += 1

        elif pos == 'SG':
            if ppg_change >= 3.5:
                criteria_met += 1
            if fg_change >= 1.8:
                criteria_met += 1
            if threepa_change >= 1.5:
                criteria_met += 1
            if ts_change >= 2.5:
                criteria_met += 1
            if ast_change >= 1.3:
                criteria_met += 1
            if ws_change >= 1.8:
                criteria_met += 1
            if mp_change >= 5.0:
                criteria_met += 1
            if tov_change <= 0.7:
                criteria_met += 1

        elif pos == 'SF':
            if ppg_change >= 3.5:
                criteria_met += 1
            if trb_change >= 1.8:
                criteria_met += 1
            if ast_change >= 1.5:
                criteria_met += 1
            if fg_change >= 1.5:
                criteria_met += 1
            if ts_change >= 2.0:
                criteria_met += 1
            if stl_change >= 0.4:
                criteria_met += 1
            if ws_change >= 2.0:
                criteria_met += 1
            if mp_change >= 5.0:
                criteria_met += 1

        elif pos == 'PF':
            if ppg_change >= 4.5:
                criteria_met += 1
            if trb_change >= 2.0:
                criteria_met += 1
            if blk_change >= 0.6:
                criteria_met += 1
            if fg_change >= 1.8:
                criteria_met += 1
            if ts_change >= 2.5:
                criteria_met += 1
            if ws_change >= 2.0:
                criteria_met += 1
            if mp_change >= 5.5:
                criteria_met += 1
            if tov_change <= 0.7:
                criteria_met += 1

        elif pos == 'C':
            if trb_change >= 2.5:
                criteria_met += 1
            if blk_change >= 0.7:
                criteria_met += 1
            if fg_change >= 2.0:
                criteria_met += 1
            if ppg_change >= 3.5:
                criteria_met += 1
            if ts_change >= 2.5:
                criteria_met += 1
            if ws48_change >= 0.03:
                criteria_met += 1
            if mp_change >= 5.0:
                criteria_met += 1
            if tov_change <= 0.4:
                criteria_met += 1


        if criteria_met >= 4:
            df.loc[i, 'breakout'] = 1

    return df

# Convert per-game statistics to float
columns_to_convert = ['MP/G', 'TRB/G', 'AST/G', 'STL/G', 'BLK/G', 'TOV/G', 'PTS/G']
df_grouped[columns_to_convert] = df_grouped[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Apply the breakout probability function to the grouped dataframe
df_grouped = define_breakout(df_grouped)


pd.set_option('display.max_rows', None)

# Replace NaN or empty values in all feature columns with 0
df_grouped.fillna(0, inplace=True)

# Print the total number of breakout seasons
print(f"Total number of breakout seasons: {len(df_grouped[df_grouped['breakout'] == 1])}")

# Print the total number of unique players
print(f"Total number of unique players: {len(df_grouped['Player'].unique())}")

# Print the total number of rows in the dataframe
print(f"Total number of rows in the dataframe: {len(df_grouped)}")

# Print the number of breakout seasons for specific years
for year in range(2009, 2018):  # Loop through years 2009 to 2017
    breakout_count = len(df_grouped[(df_grouped['breakout'] == 1) & (df_grouped['Year'] == float(year))])
    print(f"Number of breakout seasons in {year}: {breakout_count}")
