import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class LeaguesDataProcessor:
    def __init__(self):
        self.leagues_df = self.get_leagues_data()
        self.label_enc, self.leagues_ohe = self.get_leagues_encoder()

    def get_leagues_data(self):
        # Read in leagues data
        with open("leagues.json", "r") as json_file:
            leagues_data = json.load(json_file)

        # Iterate through leagues_data and create a dataframe with the league_id, league_name, and league_region
        leagues_df = []
        for league in leagues_data:
            league_id = league['id']
            league_name = league['name']
            league_region = league['region']
            league_tournaments = [x['id'] for x in league['tournaments']]
            league_df_sub = pd.DataFrame(
                {'league_id': league_id, 'league_name': league_name, 'league_region': league_region,
                 'league_tournaments': league_tournaments})
            leagues_df.append(league_df_sub)
        return pd.concat(leagues_df)

    def get_leagues_encoder(self):
        # Create a label encoder and one hot encoder for the different league names (e.g. LCS, LEC, LCK, etc.)
        unique_leagues = np.sort(np.unique(self.leagues_df['league_name'])).tolist()
        label_enc = LabelEncoder().fit(unique_leagues)
        leagues_ohe = OneHotEncoder(handle_unknown='ignore').fit(label_enc.transform(unique_leagues).reshape(-1, 1))
        return label_enc, leagues_ohe

    def transform_league_col(self, league_col):
        # Transform a column of league names into a one hot encoded dataframe
        # After getting the dataframe append it to the original set of features using pd.concat
        return pd.DataFrame(self.leagues_ohe.transform(self.label_enc.transform(league_col).reshape(-1, 1)).toarray(),
                            columns=['League_' + x for x in self.label_enc.classes_])