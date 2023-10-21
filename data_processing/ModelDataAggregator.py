import pandas as pd
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import numpy as np
import os
from tqdm.notebook import tqdm
from data_processing.utils.download_functions import *
from copy import deepcopy

class ModelDataAggregator:
    """
    Class to aggregate the data from the different years into one dataframe
    """
    def __init__(self):
        # SET CONSTANTS FOR DATA PROCESSING
        self._window_size = 20
        self._ewm_alpha = 0.05
        self._ma_min_periods = 1

        self.league_indicators_to_drop = ['League_TFT Rising Legends', 'League_All-Star Event', 'League_MSI', 'League_Worlds', 'League_EMEA Masters']
        self.non_game_features = ['platformGameId', 'esportsGameId', 'team_id', 'start_time', 'tournament_name']
        self.league_indicators = ['League_Arabian League','League_CBLOL','League_CBLOL Academy','League_College Championship','League_Elite Series','League_Esports Balkan League',
                         'League_Greek Legends League','League_Hitpoint Masters','League_LCK','League_LCK Academy','League_LCK Challengers','League_LCL','League_LCO','League_LCS',
                         'League_LCS Challengers','League_LCS Challengers Qualifiers','League_LEC','League_LJL','League_LJL Academy','League_LLA','League_LPL','League_La Ligue FranÃ§aise',
                         'League_Liga Portuguesa','League_NLC','League_North Regional League','League_PCS','League_PG Nationals','League_Prime League',
                         'League_South Regional League','League_SuperLiga','League_TCL','League_Ultraliga','League_VCS']
        # Include a set of features that are known to be important for the model 
        self.mandatory_features = ['outcome', 'outcome_domestic', 'outcome_international', 'domestic_game_ind', 
                                   'eSportsLeague_1', 'eSportsLeague_2', 'eliteLeague_1', 'eliteLeague_2', 'majorLeague_1', 'majorLeague_2', 'year']
        # Mark special features that require a different style of processing 
        self.special_features = ['outcome_domestic', 'outcome_international']

        self. important_features = ['team_1', 'team_2', 'outcome', 'outcome_domestic', 'outcome_international', 'eSportsLeague_1', 'eSportsLeague_2', 
                      'team_share_of_totalGold_at_20', 'team_share_of_totalGold_at_game_end', 
                      'team_share_of_towerKills_at_20', 'team_share_of_towerKills_at_game_end', 'team_share_of_VISION_SCORE_at_game_end']

        
        # Maintain a manual dictionary of team_id to league_indicator. This is necessary to mark the regions for teams in international tournaments (MSI/worlds since LPL teams don't have data)
        # Loop through this and mark the league_indicator for each team_id as =1 for the rows where the team_id is present
        self.league_indicator_dict = {
            98767991954244555: 'League_VCS', # GAM
            107251245690956393: 'League_VCS', # SAIGON BUFFALOS
            98767991892579754: 'League_LPL',  # RNG
            104367068120825486: 'League_PCS',  # PSG Talon
            98767991882270868: 'League_LPL',  # EDG
            99566404850008779: 'League_LPL',  # LNG
            99566404855553726: 'League_LPL',  # FPX
            99566404852189289: 'League_LPL',  # JDG
            99566404854685458: 'League_LPL',  # TES
            105520788833075738: 'League_Elite Series', # KV Mechelen
            105520824521753126: 'League_NLC', # PSV Esports
            105543843212923183: 'League_Ultraliga', # Goskilla
            105548000936710641: 'League_Ultraliga', # Method2Madness
            103935642731826448: 'League_Elite Series', # Sector One
            104710682193583854: 'League_Ultraliga', # Topo Centras Iron Wolves
            105520822049210915: 'League_Elite Series', # Team mCon
            106334794714373670: 'League_Ultraliga', # Goexanimo
        }
        
        # Read in teams data
        with open("teams.json", "r") as json_file:
           teams_data = json.load(json_file)
        teams_dict = {}
        for team in teams_data:
            teams_dict[team['team_id']] = team['name']
        self.teams_dict = teams_dict
        
    def get_featurized_data(self, folder_paths, years):
        """
        We do the following steps to process the game data
        1) Read in the tournament rows data, which specifies the match ID, the participating teams, and the winner of the match 
        2) Read in the game rows data, which contains all the granular information about each game 
        3) Additionally process the game rows data
            i) Sort the game rows by team_id and start_time
            ii) Create features based on the stats of the team over historical games 
            iii) Handle the league region indicators for each time (as 'eSportLeague')
            iv) 
        :param folder_paths: list of strings specifying the folder paths
        :param years: list of strings specifying the years 
        :return: 
            model_data - dataframe containing the diff between the two teams for each game used for training
            processed_game_data_inf_inf_inf - dataframe containing the processed game data for each individual team used for inference
        """
        tournament_rows = pd.DataFrame()
        game_rows = pd.DataFrame()
        for (folder_path, year) in zip(folder_paths, years):
            file_names = os.listdir(folder_path)

            # Get the unique tournament names by stripping out '_game_rows.csv' and '_tournament_rows.csv'
            unique_tournament_names = [file_name.split('_game_rows.csv')[0] for file_name in file_names]
            unique_tournament_names = [x.replace('_tournament_rows.csv', '') for x in unique_tournament_names]
            unique_tournament_names = list(set(unique_tournament_names))

            # Aggregate all the game rows into one dataframe, start with an empty dataframe and append onto it to save memory
            
            for tournament_name in tqdm(unique_tournament_names):
                df_tmp = pd.read_csv(f'{folder_path}/' + tournament_name + '_tournament_rows.csv')
                # Add a column to indicate the tournament name
                df_tmp['tournament_name'] = tournament_name
                tournament_rows = pd.concat([tournament_rows, df_tmp])
            print("Tournament rows shape: ", tournament_rows.shape)

            
            for tournament_name in tqdm(unique_tournament_names):
                df_tmp = pd.read_csv(f'{folder_path}/' + tournament_name + '_game_rows.csv', index_col=0)
                # Add a column to indicate the tournament name
                df_tmp['tournament_name'] = tournament_name
                df_tmp['year'] = year
                game_rows = pd.concat([game_rows, df_tmp])
            print("Game rows shape: ", game_rows.shape)

        print("Completed data loading")
        print("Tourament rows shape: ", tournament_rows.shape)
        print("Game rows shape: ", game_rows.shape)

        game_rows = game_rows.drop(columns=self.league_indicators_to_drop, axis=1)
        game_features = [x for x in game_rows.columns if x not in self.non_game_features + self.league_indicators + self.special_features + ['year']]
        self.game_features = game_features
        
        # Get a set of all team IDs as we will iterate through them to generate the row data for each team 
        all_team_ids = np.unique(game_rows['team_id'])
        processed_game_data = self.featurize_game_rows(game_rows, all_team_ids)
        processed_game_data = self.refine_league_indicator_data(processed_game_data)
        self.game_features = game_features + self.special_features

        valid_games = self.get_valid_game_rows(tournament_rows, processed_game_data)
        model_data = self.get_model_data(valid_games, processed_game_data)
        return model_data, processed_game_data
    
    def featurize_game_rows(self, game_rows, all_team_ids, averaging_method='ewm'):
        """
        averaging_method must be either 'ewm' or 'mean'
        """
        processed_game_data = []
        for team in tqdm(all_team_ids):
            team_data = game_rows[game_rows['team_id']==team].reset_index()
            team_data = team_data.sort_values(by=['start_time'])
            team_data['num_prev_games'] = np.arange(len(team_data))
            team_data['outcome_domestic'] = np.nan
            team_data['outcome_international'] = np.nan
            # Set outcome_international for worlds and msi tournaments
            team_data.loc[team_data['tournament_name'].str.contains('worlds|msi'), 'outcome_international'] = team_data['outcome']
            # Set outcome_domestic for non-worlds and non-msi tournaments
            team_data.loc[~team_data['tournament_name'].str.contains('worlds|msi'), 'outcome_domestic'] = team_data['outcome']
            
            # First lag by 1 game so that the current game is not included in the average. Then take the mean as the trailing average 
            if averaging_method == 'ewm':
                team_data_features = team_data[self.game_features + self.special_features].shift(1).ewm(alpha=self._ewm_alpha, min_periods=self._ma_min_periods, ignore_na=False).mean()
            elif averaging_method == 'mean':
                team_data_features = team_data[self.game_features + self.special_features].shift(1).rolling(window=self._window_size, min_periods=1).mean()
            else:
                raise ValueError('averaging_method must be either "ewm" or "mean"')
            
            team_data[self.game_features + self.special_features] = team_data_features 
            # Drop rows where num_prev_games == 0 as this indicates that it's the team's first game 
            team_data = team_data[team_data['num_prev_games']!=0]
        
            # Add the team name to the dataframe
            try:
                team_name = self.teams_dict[str(team)]
            except KeyError:
                team_name = "NULL"
            # Add a column for the team name
            team_data['team_name'] = team_name
        
            # Determine the team's primary league
            team_league = team_data[self.league_indicators].mean(axis=0).idxmax()
            # Determine if it's a valid league (otherwise it'll just mark the first one)
            team_league_check = team_data[team_league].sum() > 0 # If false, then do not mark based on history, have to manually mark 
            
            # check if there are any rows where the team does not have a league_indicator (i.e., np.sum(team_data[league_indicators]) == 0) and if so, mark the team_league as 1 for those rows
            # This happens when a team plays in international tournaments 
            if team_league_check:
                team_data.loc[np.sum(team_data[self.league_indicators], axis=1)==0, team_league] = 1
            else:
                pass
        
            # update the processed_game_data with the new league_indicator values
            processed_game_data.append(team_data)
        
        del game_rows  # Don't need this anymore once we're done processing them
        
        processed_game_data = pd.concat(processed_game_data)
        processed_game_data.drop('index', axis=1, inplace=True)
        return processed_game_data
    
    def refine_league_indicator_data(self, processed_game_data):
        # Next deal with marking specific team's leagues 
        for team_id, league_indicator in self.league_indicator_dict.items():
            processed_game_data.loc[processed_game_data['team_id']==team_id, league_indicator] = 1
            
        # Create two additional features related to the esport league
        # If the team is an LPL or LCK team, mark indicator 'eliteLeague' as 1
        # If the team is an LEC, LCS, LPL, or LCK team, mark indicator 'majorLeague' as 1
        processed_game_data['eliteLeague'] = (processed_game_data['League_LPL'] == 1) | (processed_game_data['League_LCK'] == 1)
        processed_game_data['majorLeague'] = (processed_game_data['League_LPL'] == 1) | (processed_game_data['League_LCK'] == 1) | \
                                             (processed_game_data['League_LCS'] == 1) | (processed_game_data['League_LEC'] == 1)
        
        # Check that the one-hot encoding worked correctly
        if (np.sum(processed_game_data[self.league_indicators], axis=1) == 1).all():
            # Convert the league_indicator one-hot encoded columns to categorical variables
            for league in [x.replace('League_', '') for x in self.league_indicators]:
                processed_game_data['League_' + league] = processed_game_data['League_' + league].apply(lambda x: league if x==1 else '')
            # Combine it into a single column
            processed_game_data['eSportLeague'] = processed_game_data[self.league_indicators].apply(lambda x: ''.join(x), axis=1)
            processed_game_data = processed_game_data.drop(columns=self.league_indicators, axis=1)
            # Convert it to a categorical variable
            processed_game_data['eSportLeague'] = processed_game_data['eSportLeague'].astype('category')
        else:
            raise ValueError('One-hot encoding of league_indicators did not work correctly')
        return processed_game_data
        
    def get_valid_game_rows(self, tournament_rows, processed_game_data):
        valid_games = tournament_rows.merge(processed_game_data[['esportsGameId', 'team_id']], how='inner', 
                                    left_on=['esportsGameId', 'team_id_1'], 
                                    right_on=['esportsGameId', 'team_id'], 
                                    suffixes=['_to_drop','_to_drop'])
        valid_games = valid_games.merge(processed_game_data[['esportsGameId', 'team_id']], how='inner', 
                                            left_on=['esportsGameId', 'team_id_2'], 
                                            right_on=['esportsGameId', 'team_id'],
                                            suffixes=['_to_drop','_to_drop'])
        valid_games.drop([x for x in valid_games.columns if '_to_drop' in x], axis=1, inplace=True)
        return valid_games 
        
    def get_model_data(self, valid_games, processed_game_data):
        # Merge processed_game_data with tournament_rows for team 1
        team_1_data = valid_games.merge(processed_game_data, how='inner', 
                                            left_on=['esportsGameId', 'team_id_1'], 
                                            right_on=['esportsGameId', 'team_id'],
                                            suffixes=['_to_drop','_to_drop'])
        
        # Merge processed_game_data with tournament_rows for team 2
        team_2_data = valid_games.merge(processed_game_data, how='inner', 
                                            left_on=['esportsGameId', 'team_id_2'], 
                                            right_on=['esportsGameId', 'team_id'],
                                            suffixes=['_to_drop','_to_drop'])
        
        # Calculate the difference between the two teams for esportsGameId and for each feature
        check_esportsGameId = np.all(team_1_data['esportsGameId'] == team_2_data['esportsGameId'])
        check_team1_id = np.all(team_1_data['team_id_1'] == team_2_data['team_id_1'])
        check_team2_id = np.all(team_1_data['team_id_2'] == team_2_data['team_id_2'])
        
        if check_esportsGameId and check_team1_id and check_team2_id:
            # Calculate the difference between the two teams for each feature
            difference_data = team_1_data[self.game_features].subtract(team_2_data[self.game_features])
            # Apply special logic for computing the difference for the "outcome_domestic" and "outcome_international" features 
            for feature in self.special_features:
                # Calculate the difference between columns A and B
                diff = team_1_data[feature].fillna(0).sub(team_2_data[feature].fillna(0)) 
                # If both columns are nan then mark the difference as nan
                diff[(team_1_data[feature].isna()) & (team_2_data[feature].isna())] = np.nan
        else:
            raise Exception('esportsGameId is not the same for the two teams')
        
        self.difference_data = difference_data.columns
        # Add the difference data to the tournament_rows dataframe as well as the league data for each team
        training_data = deepcopy(valid_games)
        training_data = pd.concat([training_data.reset_index(), difference_data], axis=1)
        training_data['eSportsLeague_1'] = team_1_data['eSportLeague']
        training_data['eSportsLeague_2'] = team_2_data['eSportLeague']
        training_data['domestic_game_ind'] = training_data['eSportsLeague_1'] == training_data['eSportsLeague_2'] 
        training_data['eliteLeague_1'] = team_1_data['eliteLeague']
        training_data['eliteLeague_2'] = team_2_data['eliteLeague']
        training_data['majorLeague_1'] = team_1_data['majorLeague']
        training_data['majorLeague_2'] = team_2_data['majorLeague']
        training_data['team_1'] = team_1_data['team_name']
        training_data['team_2'] = team_2_data['team_name']
        training_data['start_time'] = team_1_data['start_time']
        training_data['year'] = team_1_data['year']
        
        # Drop the columns that were used for joining (have '_to_drop' suffix). 
        training_data.drop([x for x in training_data.columns if '_to_drop' in x] + ['index'], axis=1, inplace=True)
        
        # drop the games where the outcome is NaN, those games are when one team has not had any games yet
        training_data.dropna(subset=['outcome'], inplace=True)
        
        del team_1_data, team_2_data, difference_data
        return training_data