import pandas as pd
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import numpy as np
from tqdm.notebook import tqdm
from copy import deepcopy
import fasttreeshap
from data_processing.utils.serialrank import SerialRank
# Data utils 
import pickle
import json
import boto3
import pickle
from botocore.handlers import disable_signing
from data_processing.utils.util_functions import find_closest_key
from datetime import datetime
from catboost import CatBoostClassifier

"""
Contains three classes
1. TournamentInferenceDataProcessor: This class processes all tournaments to prepare team_id lists based on the tournament id and stage and is necessary for tournament inference
    - This class was run to generate the tournament data, which is saved in tournament_lookup.pkl and tournament_id_lookup.pkl
    - There is no need to run this class again unless there is new tournament data
2. InferenceDataGenerator: This class generates inference data compatible for the model based on the game data, model features, team_id, and time_limit
3. InferencePipeline: This class is the main class that holds the get_team_rankings, get_tournament_rankings, and get_global_rankings methods, which returns the rankings and explanations
"""


class TournamentInferenceDataProcessor:
    "This class processes all tournaments to prepare team_id lists based on the tournament id and stage"
    def __init__(self) -> None:
        with open("tournaments.json", "r") as json_file:
            tournament_data_all = json.load(json_file)   
        self.tournament_data_all = tournament_data_all

        # Organized as [tournament_id][stage_id]: [team_ids]
        self.tournament_lookup = {}
        self.tournament_id_lookup = {}
    def process_tournament_data(self):
        for tournament in tqdm(self.tournament_data_all):
            tournament_id = tournament['id']
            tournament_name = tournament['slug']
            date_obj = datetime.strptime(tournament['startDate'], '%Y-%m-%d')
            tournament_start_date = date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')
            self.tournament_id_lookup[tournament_name] = tournament_id

            self.tournament_lookup[tournament_id] = {}
            for stage in tournament['stages']:
                match_data = []
                stage_id = stage['name']
                self.tournament_lookup[tournament_id][stage_id] = {}
                for section in stage['sections']:
                    for match in section['matches']:
                        try:
                            match_data.append(self.get_game_data_full(match))
                        except ValueError:  # There are some tournament matches that are recorded, but don't get played for some reason
                            pass  # Skip all such matches
                match_data = pd.concat(match_data, ignore_index=True)
                team_ids = [int(x) for x in np.unique(np.concatenate([match_data['team_id_1'], match_data['team_id_2']])).tolist()]
                self.tournament_lookup[tournament_id][stage_id].update({'team_ids': team_ids})
                self.tournament_lookup[tournament_id][stage_id].update({'start_time': tournament_start_date})

    
    def get_game_data_full(self, games_data):
        # Iterate through t events of the match (could consist of one or many games)
        # This is called at the match level (i.e., tournament_data['stages'][0]['sections'][0]['matches'] )
        # Look in the ['games'][t]['id'] field to get the game ID
        # Look in the ['games'][t]['state'] field to see if the game is 'completed'
        # Look in the ['games'][t]['teams'] field to get the team IDs
        # Look in the ['games'][t]['teams'][x]['result']['outcome'] field to get the result of the game for each team
        # We technically only need the 'state' to verify completion and 'id' to fetch details of the game, but load in other fields for verification
        match_id = games_data['id']  # ID for the full match
        game_tables = []
        for game in games_data['games']:
            game_state = game['state']
            if game_state == 'completed':
                game_id = game['id']  # ID for the specific games in the match
                team_ids, team_outcomes = [], []
                for team in game['teams']:
                    team_ids.append(team['id'])
                    team_outcome = 1 if team['result']['outcome'] == 'win' else 0
                    team_outcomes.append(team_outcome)
                game_tables.append(
                    pd.DataFrame({'match_id': match_id, 'esportsGameId': game_id,
                                'team_id_1': team_ids[0], 'outcome_1': team_outcomes[0],
                                'team_id_2': team_ids[1], 'outcome_2': team_outcomes[1]}, index=[0]))
        return pd.concat(game_tables, ignore_index=True)
    
class InferenceDataGenerator:
    """
    Class to generate inference data for Tournament, Team, and Global rankings 
    """
    def __init__(self, game_data, model_features) -> None:
        self.model_features = model_features
        self.numeric_model_features = np.intersect1d(game_data._get_numeric_data().columns, model_features)
        self.special_features = ['outcome_domestic', 'outcome_international']
        self.game_data = game_data
        self.additive_boost_factors = {
            # Major regions
            'LCK': 0.3,
            'LPL': 0.3,
            'LEC': 0.20,
            'LCS': 0.15,
            # Minor international regions below
            # 'PCS': 0.05,
            # 'LLA': 0.04,
            # 'CBLOL': 0.03,
            # 'VCS': 0.02,
            # 'LJL': 0.01,
        }
        # Mark important features to boost in international competitions
        self.columns_to_boost = ['outcome_domestic', 'team_share_of_totalGold_at_game_end', 'team_share_of_towerKills_at_game_end', 'team_share_of_VISION_SCORE_at_game_end']

    def get_inference_data_by_team_id(self, team_ids, time_limit = None):
        """
        Fetches inference data for the team_ids, if time_limit is None, get the latest games. Otherwise, get the latest games that occur before the time_limit
        """
        game_data, missing_team_ids = self.get_game_data_by_team_id(team_ids, time_limit)
        for column in self.columns_to_boost:
            game_data[column] *= game_data['eSportLeague'].map(self.additive_boost_factors).fillna(0)

        # Exclude the missing team ids from the team_ids list, if there's no missing teams it returns the full list
        team_ids = np.setdiff1d(team_ids, missing_team_ids)
        tournament_rows = self.get_tournament_rows_by_team_id(team_ids)
        inference_data = self.get_inference_data(tournament_rows, game_data)
        
        return inference_data, team_ids, missing_team_ids

    def get_game_data_by_team_id(self, team_ids, time_limit = None):
        """
        Gets the last game played for each team_id 
        """
        game_data = self.game_data[self.game_data['team_id'].isin(team_ids)]
        game_data = game_data.sort_values(by=['team_id', 'start_time'])
        if time_limit is not None: 
            game_data = game_data[game_data['start_time'] < time_limit]
            game_data = game_data.drop_duplicates(subset=['team_id'], keep='last')
        else:
            game_data = game_data.drop_duplicates(subset=['team_id'], keep='last')
        self.game_data_tmp = game_data
        
        # game_data should be [N_valid_inference, n_features] but it's possible that there's a new team
        # In that case, we do the ranking without them and slot that team in the middle.
        missing_team_ids = np.setdiff1d(team_ids, game_data['team_id'])
        
        if len(missing_team_ids) > 0:
            print("WARNING: Inference includes that that previously has not played any games. This team will be ranked in the middle of the pack by default")
            print(f"Missing team ids: {missing_team_ids}")

        return game_data, missing_team_ids

    def get_tournament_rows_by_team_id(self, team_ids):
        """
        Creates an imaginary round-robin tournament
        """
        tournament_rows = []
        for team_1 in team_ids:
            for team_2 in team_ids:
                tournament_rows.append([team_1, team_2])
                    
        # Format this as a table
        tournament_rows = pd.DataFrame(tournament_rows, columns=['team_id_1', 'team_id_2'])
        return tournament_rows

    def get_inference_data(self, tournament_rows, game_data):
        tournament_rows_featurized_1 = tournament_rows.merge(game_data, how='left', left_on=['team_id_1'], right_on=['team_id'])
        tournament_rows_featurized_2 = tournament_rows.merge(game_data, how='left', left_on=['team_id_2'], right_on=['team_id'])

        # Compute the difference between the two teams (with additional checks to ensure that no shuffling occurred during the join)
        check_team1_id = np.all(tournament_rows_featurized_1['team_id_1'] == tournament_rows_featurized_2['team_id_1'])
        check_team2_id = np.all(tournament_rows_featurized_1['team_id_2'] == tournament_rows_featurized_2['team_id_2'])
        check_team1_id_base = np.all(tournament_rows['team_id_1'] == tournament_rows_featurized_1['team_id_1'])
        check_team2_id_base = np.all(tournament_rows['team_id_2'] == tournament_rows_featurized_1['team_id_2'])

        if check_team1_id and check_team2_id and check_team1_id_base and check_team2_id_base:
            # Calculate the difference between the two teams for each feature
            difference_data = tournament_rows_featurized_1[self.numeric_model_features].subtract(tournament_rows_featurized_2[self.numeric_model_features])
            for feature in self.special_features:
                # Calculate the difference between columns A and B
                diff = tournament_rows_featurized_1[feature].fillna(0).sub(tournament_rows_featurized_2[feature].fillna(0)) 
                # If both columns are nan then mark the difference as nan
                diff[(tournament_rows_featurized_1[feature].isna()) & (tournament_rows_featurized_2[feature].isna())] = np.nan
                difference_data[feature] = diff
        else:
            raise Exception('esportsGameId is not the same for the two teams')
                
        # Add the difference data to the tournament_rows dataframe as well as the league data for each team
        training_data = deepcopy(tournament_rows)
        training_data = pd.concat([training_data.reset_index(), difference_data], axis=1)
        training_data['eSportsLeague_1'] = tournament_rows_featurized_1['eSportLeague']
        training_data['eSportsLeague_2'] = tournament_rows_featurized_2['eSportLeague']
        training_data['domestic_game_ind'] = training_data['eSportsLeague_1'] == training_data['eSportsLeague_2']
        training_data['eliteLeague_1'] = tournament_rows_featurized_1['eliteLeague']
        training_data['eliteLeague_2'] = tournament_rows_featurized_2['eliteLeague']
        training_data['majorLeague_1'] = tournament_rows_featurized_1['majorLeague']
        training_data['majorLeague_2'] = tournament_rows_featurized_2['majorLeague']
        training_data['team_1'] = tournament_rows_featurized_1['team_name']
        training_data['team_2'] = tournament_rows_featurized_2['team_name']
        training_data['start_time'] = tournament_rows_featurized_1['start_time']
        training_data['year'] = tournament_rows_featurized_1['year']

        # Drop the columns that were used for joining (have '_to_drop' suffix). 
        training_data.drop([x for x in training_data.columns if '_to_drop' in x] + ['index'], axis=1, inplace=True)
        training_data = training_data.drop(['team_id_1', 'team_id_2', 'start_time'], axis=1)

        return training_data[self.model_features]
    
    
class InferencePipeline:
    def __init__(self) -> None:
        # Set up AWS S3 reader
        s3 = boto3.resource('s3')
        s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
        self.aws_bucket_name = "leaguelflegendsglobalpowerrankings"
        self.game_data_filename = "game_data.csv.gz"
        self.teams_dict_filename = 'teams_dict.pkl'
        self.tournaments_filename = 'tournaments.pickle'
        self.tournament_lookup_filename = 'tournament_lookup.pkl'
        self.tournament_id_lookup_filename = 'tournament_id_lookup.pkl'
        self.model_features_filename = 'model_features_dict.pkl'
        self.catboost_model_filename = 'catboost_model.cbm'

        print('Loading data from S3...')
        print('Loading metadata files...')
        with open(self.teams_dict_filename, 'wb') as data:
            s3.Bucket(self.aws_bucket_name).download_fileobj(self.teams_dict_filename, data)
        with open(self.teams_dict_filename, 'rb') as data:
            self.teams_dict = pickle.load(data) 
        self.teams_dict_rev = {v:k for (k,v) in self.teams_dict.items()}

        with open(self.tournament_lookup_filename, 'wb') as data:
            s3.Bucket(self.aws_bucket_name).download_fileobj(self.tournament_lookup_filename, data)
        with open(self.tournament_lookup_filename, 'rb') as data:
            self.tournament_lookup = pickle.load(data)
        
        with open(self.tournament_id_lookup_filename, 'wb') as data:
            s3.Bucket(self.aws_bucket_name).download_fileobj(self.tournament_id_lookup_filename, data)
        with open(self.tournament_id_lookup_filename, 'rb') as data:
            self.tournament_id_lookup = pickle.load(data)

        with open(self.model_features_filename, 'wb') as data:
            s3.Bucket(self.aws_bucket_name).download_fileobj(self.model_features_filename, data)
        with open(self.model_features_filename, 'rb') as data:
            self.model_features_dict = pickle.load(data)
        # Use the values of the dict to map the feature explanations 
        self.model_features = list(self.model_features_dict.keys())

        with open(self.tournaments_filename, 'wb') as data:
            s3.Bucket(self.aws_bucket_name).download_fileobj(self.tournaments_filename, data)
        with open(self.tournaments_filename, 'rb') as data:
            self.tournament_data_all = pickle.load(data)

        print('Loading game data...')
        with open(self.game_data_filename, 'wb') as data:
            s3.Bucket(self.aws_bucket_name).download_fileobj(self.game_data_filename, data)
        self.game_data = pd.read_csv(self.game_data_filename, compression='gzip')

        # Create inference data generator
        self.inference_data_generator = InferenceDataGenerator(self.game_data, self.model_features)

        # Load in pretrained model
        print('Loading model...')
        with open( self.catboost_model_filename, 'wb') as data:
            s3.Bucket(self.aws_bucket_name).download_fileobj(self.catboost_model_filename, data)
        self.model = CatBoostClassifier().load_model("catboost_model.cbm")
        self.shap_explainer = fasttreeshap.TreeExplainer(self.model, algorithm = "v1", n_jobs = -1)

        self.shap_features_to_exclude = ['team_1', 'team_2', 'outcome_domestic', 'outcome_international', 'domestic_game_ind', 'eliteLeague_1', 'eliteLeague_2', 
            'majorLeague_1', 'majorLeague_2', 'eSportsLeague_1', 'eSportsLeague_2', 'year', 'team_share_of_totalGold_at_20', 'team_share_of_totalGold_at_game_end',
            'team_share_of_towerKills_at_20', 'team_share_of_towerKills_at_game_end', 'team_share_of_VISION_SCORE_at_game_end', 'support_NEUTRAL_MINIONS_KILLED_at_30']

        ## Load in all data required for inference (this loads locally - not from S3)
        # with open("tournaments.json", "r") as json_file:
        #     self.tournament_data_all = json.load(json_file)

        # # Load both tournament_lookup and tournament_id_lookup using pickle
        # with open('tournament_lookup.pkl', 'rb') as f:
        #     self.tournament_lookup = pickle.load(f)

        # with open('tournament_id_lookup.pkl', 'rb') as f:
        #     self.tournament_id_lookup = pickle.load(f)

        # with open('teams_dict.pkl', 'rb') as f:
        #     self.teams_dict = pickle.load(f)

        # self.teams_dict_rev = {v:k for (k,v) in self.teams_dict.items()}

        # # Load in everything needed for inference
        # with open("model_features.txt", "rb") as fp:   # Unpickling
        #     self.model_features = pickle.load(fp)

        # self.game_data = pd.read_csv('game_data.csv')

        # # Create inference data generator
        # self.inference_data_generator = InferenceDataGenerator(self.game_data, self.model_features)

        # # Load in pretrained model
        # self.model = CatBoostClassifier().load_model("catboost_model.cbm")
        # self.shap_explainer = fasttreeshap.TreeExplainer(self.model, algorithm = "v1", n_jobs = -1)

    def find_team_id(self, team_name, approximate_match = True):
        if approximate_match:
            return int(find_closest_key(team_name, self.teams_dict_rev))
        else:
            return int(self.teams_dict_rev[team_name])

    def get_tournament_rankings(self, tournament_id, stage_name):
        team_ids = self.tournament_lookup[tournament_id][stage_name]['team_ids']
        start_time = self.tournament_lookup[tournament_id][stage_name]['start_time']
        
        X_inference, valid_team_ids, missing_team_ids = self.inference_data_generator.get_inference_data_by_team_id(team_ids, start_time)
        team_ranks = self.run_inference_get_rankings(X_inference, valid_team_ids, missing_team_ids)
        team_top_features = self.explain_inference(X_inference, valid_team_ids, missing_team_ids)
        return team_ranks, team_top_features

    def get_global_rankings(self, k):
        team_ids = self.game_data['team_id'].unique()
        # Exclude team_ids if they are not identifiable in the teams_dict
        team_ids = [x for x in team_ids if str(x) in self.teams_dict.keys()]
        X_inference, valid_team_ids, missing_team_ids = self.inference_data_generator.get_inference_data_by_team_id(team_ids)
        # Fill X_inference with 0's for the missing data
        X_inference = X_inference.fillna(0)
        team_ranks = self.run_inference_get_rankings(X_inference, valid_team_ids, missing_team_ids)
        # Filter team_ranks to the top k
        team_ranks = team_ranks.iloc[:k]
        # Get new team ids and run the computation again to shorter inference time for explainations
        team_ids = [int(self.teams_dict_rev[x]) for x in team_ranks['team_name'].tolist()]
        X_inference, valid_team_ids, missing_team_ids = self.inference_data_generator.get_inference_data_by_team_id(team_ids)

        team_top_features = self.explain_inference(X_inference, valid_team_ids, missing_team_ids)
        # Keep only explainations for the top k teams
        team_top_features = team_top_features.loc[team_ranks['team_name'].tolist()]
        return team_ranks, team_top_features

    def get_team_rankings(self, team_ids):
        X_inference, valid_team_ids, missing_team_ids = self.inference_data_generator.get_inference_data_by_team_id(team_ids)
        team_ranks = self.run_inference_get_rankings(X_inference, valid_team_ids, missing_team_ids)
        team_top_features = self.explain_inference(X_inference, valid_team_ids, missing_team_ids)
        return team_ranks, team_top_features

    def run_inference_get_rankings(self, X_inference, valid_team_ids, missing_team_ids, probabilistic = True, ranking_method = 'probabilistic'):
        # Prediction creates an N x N matrix where N is the len(unique_team_ids)
        # Then based on the specified method, it returns the rankings
        # ranking_method must be ['probabilistic', 'spectral']

        ## Probabilistic prediction
        if probabilistic:
            tournament_preds = self.model.predict_proba(X_inference)[:,1]
            # Format the predictions into an N x N matrix (fill by row) 
            tournament_preds = tournament_preds.reshape(len(valid_team_ids), len(valid_team_ids))
            # Multiply the values less than 0.5 by -1 to get the correct sign
            tournament_preds[tournament_preds < 0.5] = tournament_preds[tournament_preds < 0.5]*-1
            
        else:
            tournament_preds = self.model.predict(X_inference)
            # Format the predictions into an N x N matrix (fill by row) 
            tournament_preds = tournament_preds.reshape(len(valid_team_ids), len(valid_team_ids))
            # Convert the 0's into -1 (means they lost)
            tournament_preds[tournament_preds==0] = -1
        
        # Mark the diagonal as 0's
        np.fill_diagonal(tournament_preds, 0.0)

        if ranking_method == 'spectral':
            serial_rank = SerialRank(tournament_preds)
            serial_rank.fit()
            team_scores = serial_rank.r.squeeze()
        elif ranking_method == 'probabilistic':
            team_scores = np.sum(tournament_preds, axis=1)

        # Rank the teams based on the scores assigned by the specified method 
        team_ranks = pd.DataFrame({'team_name': [self.teams_dict[str(x)] for x in valid_team_ids], 'score': team_scores}).sort_values(by='score', ascending=False)
        missing_teams = pd.DataFrame({'team_name': [self.teams_dict[str(x)] for x in missing_team_ids], 'score': np.median(team_ranks['score'])-1e-5})
        team_ranks = pd.concat([team_ranks, missing_teams], axis=0).reset_index(drop=True).sort_values(by='score', ascending=False)
        
        # Returns a N x 2 dataframe with the team_name and score
        return team_ranks
    

    def explain_inference(self, X_inference, valid_team_ids, missing_team_ids):
        # Take the (column-wise) mean of the shap_values for each n-sized block of rows (representing the games that a specific team plays)
        # Skip the first row of each block since that's the difference between the first team and itself

        shap_values = self.shap_explainer(X_inference)
        N_valid_inference = len(valid_team_ids)
        for i in range(0, len(shap_values), N_valid_inference):
            if i == 0:
                team_shap_values = shap_values[i+1:i+N_valid_inference].values.mean(axis=0)
            else:
                team_shap_values = np.vstack((team_shap_values, shap_values[i+1:i+N_valid_inference].values.mean(axis=0)))

        # Label the shap values with the team names and feature names
        team_shap_values = pd.DataFrame(team_shap_values, columns=X_inference.columns, index=[self.teams_dict[str(x)] for x in valid_team_ids])

        # Drop the features that are in the shap_features_to_exclude list (not interpretable)
        team_shap_values = team_shap_values.drop(self.shap_features_to_exclude, axis=1)

        # For each team, get the top 5 most positive features and top 5 most negative features and put them into the columns ['top_1_pos', 'top_2_pos', ..., 'top_1_neg', 'top_2_neg', ..., 'top_5_neg']
        # Define a custom function to get the top 5 most positive and negative features for a team
        def get_top_features(row):
            top_pos = row.sort_values(ascending=False)[:5].index.tolist()
            top_neg = row.sort_values(ascending=True)[:5].index.tolist()
            return pd.Series(top_pos + top_neg, index=[f'top_{i}_pos' for i in range(1, 6)] + [f'top_{i}_neg' for i in range(1, 6)])

        # Apply the custom function to each row of the team_shap_values dataframe
        team_top_features = team_shap_values.apply(get_top_features, axis=1)

        # Create a dataframe with the missing teams
        missing_teams = pd.DataFrame(index=[self.teams_dict[str(x)] for x in missing_team_ids], columns=team_top_features.columns)

        # Concatenate the missing teams dataframe with the original team top features dataframe
        team_top_features = pd.concat([team_top_features, missing_teams])

        # Use the values of the model_features_dict to map the feature explanations
        team_top_features_mapped = team_top_features.copy()

        # Loop over the keys and values of the model_features_dict
        for key, value in self.model_features_dict.items():
            # Replace all occurrences of the key with the value in the team_top_features_mapped dataframe
            team_top_features_mapped = team_top_features_mapped.replace({key: value})

        # Returns the mapped team_top_features dataframe (N x 10 dataframe with the top 5 positive and top 5 negative features for each team
        return team_top_features_mapped


    
