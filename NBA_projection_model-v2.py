#!/usr/bin/env python
# coding: utf-8

# # Fantasy Projection Model Using NBA Per Game Stats

# import modules
import numpy as np
import pandas as pd
from sklearn import preprocessing
from math import sqrt
from IPython.display import display
from sklearn.metrics import mean_squared_error
import pprint

# read in per game data from csv folder
original_df = pd.read_csv('nba-csv/player_general_traditional_per_game_data.csv', header=0)

# check to see what data looks like
original_df.tail()

# filter for players who played atleast 9 games
gp_filter = original_df['gp'] > 9
df1 = original_df[gp_filter]

# function to normalize data
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

# apply normalize function for each column
def vorp(df):
    for col_name in cols_to_norm:
        df['{}_norm'.format(col_name)] = normalize(df[col_name])
    return df

cols_to_norm = ['pts',
                'min',
               'fgm',
               'fga',
               'fg3m',
               'fg3a',
               'ftm',
               'fta',
               'oreb',
               'dreb',
               'ast',
               'stl',
               'tov',
               'blk']

# normalize data be season averages and league totals
df = df1.groupby(['season_id']).apply(vorp)

# season_list for NBA players
season_list = [
    '1996-97',
    '1997-98',
    '1998-99',
    '1999-00',
    '2000-01',
    '2001-02',
    '2002-03',
    '2003-04',
    '2004-05',
    '2005-06',
    '2006-07',
    '2007-08',
    '2008-09',
    '2009-10',
    '2010-11',
    '2011-12',
    '2012-13',
    '2013-14',
    '2014-15',
    '2015-16',
    '2016-17'
]

# function to calculate distance between two points
def calc_distance(u, v):
    dist = np.sqrt(np.sum((u - v)**2))
    return dist

# create a function to find the player and the next season
def find_player(player_id, season):
    # replaces for loop
    for row in df.itertuples():
        if season == row.season_id and player_id == row.player_id:
            return row

def player_comparison_tool(current_player_season, current_player_id):
    for row in df.itertuples():
        if current_player_season == row.season_id and current_player_id == row.player_id:
            current_player_id = row.player_id
            break
            
    if (current_player_id == None):
        print('Can\'t find player with id {0} and season {1}'.format(current_player_id, current_player_season))
        return

    current_player_vector = np.array([
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'pts_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'min_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'fgm_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'fga_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'fg3m_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'fg3a_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'ftm_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'fta_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'oreb_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'dreb_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'ast_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'stl_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'tov_norm']).item(),
        (df.loc[(df['player_id'] == current_player_id) & (df['season_id'] == current_player_season), 'blk_norm']).item()
    ])

    print('Projecting player_id {0} for season {1}'.format(current_player_id, season_list[(season_list.index(row.season_id) + 1)]))

    # create a list to store the data
    player_distance = []

    # loop over every row in the dataframe to calculate percent error
    weighted_numbers = [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for row in df.itertuples():
        compared_player_vector = np.array([
        row.pts_norm,
        row.min_norm,
        row.fgm_norm,
        row.fga_norm,
        row.fg3m_norm,
        row.fg3a_norm,
        row.ftm_norm,
        row.fta_norm,
        row.oreb_norm,
        row.dreb_norm,
        row.ast_norm,
        row.stl_norm,
        row.tov_norm,
        row.blk_norm
        ])
        
        vfunc = np.vectorize(calc_distance)
        distance_vect = vfunc(current_player_vector, compared_player_vector)
        weighted_distance = distance_vect * weighted_numbers
        number = np.sum(weighted_distance)
        player_distance.append(number)
        
    # create a new column with error 
    df['distance'] = player_distance

    # sort dataframe by smallest distance
    ranked_df = df.sort_values('distance')
    
    stats = ['pts',
             'min',
             'fgm',
             'fga',
             'fg3m',
             'fg3a',
             'ftm',
             'fta',
             'oreb',
             'dreb',
             'ast',
             'stl',
             'tov',
             'blk'
             ]
    
    # create empty dictionary to put in projected stats
    projected_stats = {}

    for col in stats:
        sum_stat = 0
        sum_weight = 0
        for index, row in ranked_df.iloc[1:11].iterrows():
            # skip over the row if it was 2016-17 season because we can't take the next
            if row.season_id == '2016-17':
                continue
            # get the players next season
            weight = (1 / row.distance)
            next_season = season_list[(season_list.index(row.season_id) + 1)]
            # find the player row with the id and the next season
            player_next_season = find_player(row.player_id, next_season)
            sum_stat += getattr(player_next_season, col) * weight
            sum_weight += weight
        projected_stats['player_id'] = current_player_id
        projected_stats['proj_season_id'] = season_list[(season_list.index(current_player_season) + 1)]
        projected_stats['proj_' + col] = (sum_stat / sum_weight)
    return projected_stats

# small sample of player_ids
player_ids = [
    201939,
    201935,
    201142,
    202326,
    2544,
    203081,
    203076,
    201566,
    1626164,
    101150,
    200768,
    202710,
    202689,
    101108,
    203114
]


# run for loop for each player
final_projections = []
for baller_id in player_ids:
    current_player_id = baller_id
    current_player_season = '2015-16'
    # if function to catch if player is not in player dataframe, if not then don't even try the function
    try:
        projections = player_comparison_tool(current_player_season, current_player_id)
        if (projections == None):
            continue
    except:
        continue
    final_projections.append(projections)

proj_columns = [
    'player_id',
    'proj_season_id',
    'proj_pts',
    'proj_min',
    'proj_fgm',
    'proj_fga',
    'proj_fg3m',
    'proj_fg3a',
    'proj_ftm',
    'proj_fta',
    'proj_oreb',
    'proj_dreb',
    'proj_ast',
    'proj_stl',
    'proj_tov',
    'proj_blk'  
]


# convert final projections into a dataframe to split
proj_df = pd.DataFrame(columns=proj_columns, data=final_projections)


# merge dataframes on player_id column and season_ids
final_df = pd.merge(proj_df, df,  how='left', left_on=['player_id','proj_season_id'], right_on = ['player_id','season_id'])

columns_to_drop = [
    'pts_norm',
    'min_norm',
    'fgm_norm',
    'fga_norm',
    'fg3m_norm',
    'fg3a_norm',
    'ftm_norm',
    'fta_norm',
    'oreb_norm',
    'dreb_norm',
    'ast_norm',
    'stl_norm',
    'tov_norm',
    'blk_norm',
    'distance'
]


final_df.drop(columns = columns_to_drop, inplace = True)

# get player name from csv to merge with player id
player_df = pd.read_csv('nba-csv/player_name_player_id_all_seasons_final.csv')
season = player_df['season_id'] == '2016-17'
player_df = player_df[season]

player_proj = pd.merge(final_df, player_df[['player_name', 'player_id']], how = 'left', on = 'player_id').drop_duplicates().reset_index(drop=True)

player_proj.head(10)

player_info_columns = ['player_name',
                      'player_id',
                      'proj_season_id']


player_info = player_proj[player_info_columns]


# ## Measure Effectiveness with RMSE

df_real = player_proj.loc[:, ['pts','min','fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','ast','stl','tov', 'blk']]
df_proj = player_proj.loc[:, ['proj_pts','proj_min','proj_fgm','proj_fga','proj_fg3m','proj_fg3a','proj_ftm','proj_fta','proj_oreb','proj_dreb','proj_ast','proj_stl','proj_tov', 'proj_blk']]

# calculate mean square error
lin_mse = mean_squared_error(df_real, df_proj, multioutput='raw_values')
lin_rmse = np.sqrt(lin_mse)
confidence = np.mean(lin_rmse)
print('{0} percent confidence in projected {1} per game stats'.format(100 - round(confidence, 2), '2016-17'))

df_real_stats = pd.concat([player_info,df_real],axis=1)
df_real_stats.rename(columns = {'proj_season_id':'season_id',},inplace = True)

df_proj_stats = pd.concat([player_info,df_proj],axis=1)
df_proj_stats.rename(columns = {'proj_season_id':'season_id',
                               'proj_pts': 'pts',
                                'proj_min':'min',
                                'proj_fgm':'fgm',
                                'proj_fga':'fga',
                                'proj_fg3m':'fg3m',
                                'proj_fg3a':'fg3a',
                                'proj_ftm':'ftm',
                                'proj_fta':'fta',
                                'proj_oreb':'oreb',
                                'proj_dreb':'dreb',
                                'proj_ast':'ast',
                                'proj_stl':'stl',
                                'proj_tov':'tov',
                               'proj_blk':'blk'}, inplace = True)


# Print the differences for a spot check
display(df_real_stats)
display(round(df_proj_stats,1))

# # Converting stats to fantasy points
def fantasy_pts_converter(row):
    pts = row[3]
    fgm = row[5]
    fga = row[6]
    ftm = row[9]
    fta = row[10]
    oreb = row[11]
    dreb = row[12]
    ast = row[13]
    stl = row[14]
    tov = row[15]
    blk = row[16]    
    fantasy_pts = pts + fgm - fga + ftm - fta + oreb + dreb + ast + stl - tov + blk
    return fantasy_pts

fantasy_value = df_proj_stats.apply(fantasy_pts_converter, axis = 'columns')

df_proj_stats['fantasy_pts'] = fantasy_value
df_proj_stats.sort_values('fantasy_pts', ascending = False)

# # Comparing to competitors

# ### Convert current columns to similar competitor columns
columns = ['player_name',
           'player_id',
           'proj_pts',
           'proj_min',
           'proj_fgm',
           'proj_fga',
           'proj_fg3m',
           'proj_fg3a',
           'proj_ftm',
           'proj_fta',
           'proj_oreb',
           'proj_dreb',
           'proj_ast',
           'proj_stl',
           'proj_tov',
           'proj_blk']

df_proj_final = player_proj.loc[:, columns]

df_proj_final['proj_fg%'] = df_proj_final['proj_fgm']  / df_proj_final['proj_fga']
df_proj_final['proj_ft%'] = df_proj_final['proj_ftm'] + df_proj_final['proj_fta']
df_proj_final['proj_reb'] = df_proj_final['proj_oreb'] + df_proj_final['proj_dreb']

final_columns = ['player_name',
           'player_id',
           'proj_pts',
           'proj_reb',
           'proj_ast',
           'proj_blk',
           'proj_stl',
           'proj_fg%',
           'proj_ft%',
           'proj_fg3m',
           'proj_min',
           'proj_tov']

df_proj_final = df_proj_final[final_columns]

# ### Read in competitor data

# read in projections to dataframe
df_comp_1 = pd.read_csv('nba-csv/ESPN_CBS_FantasyPros_Fantasy_Basketball_Overall_2018_Average_Projections.csv')
#df_comp_2 = pd.read_csv('Hashtag_CBS_FantasyPros_Fantasy_Basketball_Overall_2018_Average_Projections.csv')

columns_to_drop = ['Team', 'Positions', 'GP']

df_comp_1.drop(columns = columns_to_drop, inplace = True)

# find names and match to player_id
lowercase_names = df_comp_1['Player'].str.lower()
df_comp_1['Player'] = lowercase_names

# merge with player_name
player_df = pd.read_csv('nba-csv/player_name_player_id_all_seasons_final.csv')
season = player_df['season_id'] == '2016-17'
player_df = player_df[season]

lowercase = player_df['player_name'].str.lower()

player_df['player_name'] = lowercase

comp_merged = pd.merge(df_comp_1, player_df[['player_name','player_id']], how = 'left', left_on = 'Player', right_on = 'player_name').drop_duplicates().reset_index(drop=True)

comp_merged.dropna(how = 'any', inplace = True)

player_ids = comp_merged['player_id'].astype(int)

comp_merged['player_id'] = player_ids

del comp_merged['Player']

cols = [
    'player_name',
    'player_id',
    'PTS',
    'REB',
    'AST',
    'BLK',
    'STL',
    'FG%',
    'FT%',
    '3PM',
    'MIN',
    'TO']

comp_merged = comp_merged[cols]

df_real = player_proj.loc[:, ['player_name', 'player_id', 'pts','min','fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','ast','stl','tov', 'blk']]

df_real['fg%'] = df_real['fgm']  / df_real['fga']
df_real['ft%'] = df_real['ftm'] + df_real['fta']
df_real['reb'] = df_real['oreb'] + df_real['dreb']

final_real_columns = ['player_name',
           'player_id',
           'pts',
           'reb',
           'ast',
           'blk',
           'stl',
           'fg%',
           'ft%',
           'fg3m',
           'min',
           'tov']

df_real_final = df_real[final_real_columns]

competitor_final = pd.merge(comp_merged, df_real_final, how = 'left', on = 'player_id')

# temp drop until I run player_comparison_tool for all players
competitor_proj = competitor_final.dropna(how = 'any')

df_real = competitor_proj.loc[:, ['pts','reb','ast','blk','stl','fg%','ft%','fg3m','min','tov']]
df_proj = competitor_proj.loc[:, ['PTS','REB','AST','BLK','STL','FG%','FT%','3PM','MIN','TO']]

# calculate mean square error
lin_mse = mean_squared_error(df_real, df_proj, multioutput='raw_values')
lin_rmse = np.sqrt(lin_mse)
confidence = np.mean(lin_rmse)
print('{0} percent confidence in projected {1} per game stats'.format(100 - round(confidence, 2), '2016-17'))


# ### Match up against our model for same stat columns

model_final = pd.merge(df_proj_final, df_real_final, how = 'left', on = 'player_id')
df_real = model_final.loc[:, ['pts','reb','ast','blk','stl','fg%','ft%','fg3m','min','tov']]
df_proj = model_final.loc[:, ['proj_pts','proj_reb','proj_ast','proj_blk','proj_stl','proj_fg%','proj_ft%','proj_fg3m','proj_min','proj_tov']]

# calculate mean square error
lin_mse = mean_squared_error(df_real, df_proj, multioutput='raw_values')
lin_rmse = np.sqrt(lin_mse)
confidence = np.mean(lin_rmse)
print('{0} percent confidence in projected {1} per game stats'.format(100 - round(confidence, 2), '2016-17'))

