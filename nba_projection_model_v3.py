#!/usr/bin/env python
# coding: utf-8

# # Fantasy Projection Model Using NBA Per Game Stats

# This model takes in a player season and finds the 10 most similar player seasons across the decades. Then using weights based on how similar the player seasons are, takes the averages of each of those players following seasons to predict our current players next season.

# In[1]:


# import modules
import numpy as np
import pandas as pd
from sklearn import preprocessing
from math import sqrt
from IPython.display import display
from sklearn.metrics import mean_squared_error
import pprint


# In[ ]:


# import custom functions
from nba_functions import normalize, vorp, calc_distance, find_player, player_comparison_tool


# In[2]:


# read in per game data from csv folder
original_df = pd.read_csv('nba-csv/player_general_traditional_per_game_data_v2.csv', header=0)


# In[3]:


# check to see what data looks like
original_df.tail()


# In[4]:


# filter for players who played atleast 9 games
gp_filter = original_df['gp'] > 9
df1 = original_df[gp_filter]


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


# In[8]:


# normalize data be season averages and league totals
df = df1.groupby(['season_id']).apply(vorp)


# In[9]:


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
    '2016-17',
    '2017-18',
    '2018-19'
]


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
#for baller_id in all_player_ids:
for baller_id in player_ids:
    current_player_id = baller_id
    current_player_season = '2017-18'
    # if function to catch if player is not in player dataframe, if not then don't even try the function
    try:
        projections = player_comparison_tool(df, current_player_season, current_player_id)
        if (projections == None):
            continue
    except:
        continue
    final_projections.append(projections)


# In[15]:


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


# In[16]:


# convert final projections into a dataframe to split
proj_df = pd.DataFrame(columns=proj_columns, data=final_projections)


# In[17]:


# merge dataframes on player_id column and season_ids
final_df = pd.merge(proj_df, df,  how='left', left_on=['player_id','proj_season_id'], right_on = ['player_id','season_id'])


# In[18]:


final_df.head(5)


# In[19]:


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


# In[20]:


final_df.drop(columns = columns_to_drop, inplace = True)


# In[21]:


final_df.head(10)


# In[22]:


# get player name from csv to merge with player id
player_df = pd.read_csv('nba-csv/player_name_player_id_all_seasons_final.csv')
season = player_df['season_id'] == '2018-19'
player_df = player_df[season]


# In[23]:


player_proj = pd.merge(final_df, player_df[['player_name', 'player_id']], how = 'left', on = 'player_id').drop_duplicates().reset_index(drop=True)


# In[24]:


player_proj.head(10)


# In[25]:


player_info_columns = ['player_name',
                      'player_id',
                      'proj_season_id']


# In[26]:


player_info = player_proj[player_info_columns]


# In[27]:


player_proj.to_csv('nba-csv/player_proj_df.csv', index=False)
player_info.to_csv('nba-csv/player_info_df.csv', index=False)


# In[ ]:


print('Done.')

