import pandas as pd

class TournamentDataProcessor:
    def __init__(self, tournament_data, leagues_data):
        # Leagues_data is a list of dictionaries with the league_id, league_name, and league_region so that we can designate tournament region
        self.tournament_data = tournament_data
        self.tournament_id = tournament_data['id']
        self.tournament_league_id = tournament_data['leagueId']
        self.tournament_name = tournament_data['slug']
        self.tournament_stages = [x['name'] for x in tournament_data['stages']]
        tournament_leagues = leagues_data[leagues_data['league_id'] == self.tournament_league_id]
        try:
            self.tournament_league = tournament_leagues['league_name'].values[0]
            self.tournament_region = tournament_leagues['league_region'].values[0]
        except IndexError:
            raise IndexError(f"League ID {self.tournament_league_id} not found in leagues data")

    def get_tournament_stages(self):
        return [[x['name'], len(x['sections'])] for x in self.tournament_data['stages']]

    def get_tournament_data(self, training_stages=[], testing_stages=[]):
        # Usually set testing stages as ['Playoffs']
        # Iterate through the training stages and aggregate the data together, do the same for the testing stages
        # If training_stages=[] and testing_stages = [], then return all stages as training
        # Return the training and testing data
        # If no training stages are specified, use all stages except the testing stages
        if len(training_stages) == 0:
            training_stages = [x['name'] for x in self.tournament_data['stages'] if x['name'] not in testing_stages]

        # Validate that the training and testing stages are valid within the tournament
        for stage in training_stages + testing_stages:
            if stage not in self.tournament_stages:
                raise ValueError(f"Stage {stage} not found in tournament {self.tournament_name}")

        self.training_stages = training_stages
        self.testing_stages = testing_stages
        training_data = []
        testing_data = []
        for stage in self.tournament_data['stages']:
            if stage['name'] in training_stages:
                for section in stage['sections']:
                    for match in section['matches']:
                        training_data.append(self.get_game_data_full(match))
            elif stage['name'] in testing_stages:
                for section in stage['sections']:
                    for match in section['matches']:
                        testing_data.append(self.get_game_data_full(match))

        training_data = pd.concat(training_data, ignore_index=True)
        if len(testing_data) > 0:
            testing_data = pd.concat(testing_data, ignore_index=True)
        else:  # Return an empty frame with the same columns as training data
            testing_data = pd.DataFrame(columns=training_data.columns)
        self.training_data = training_data
        self.testing_data = testing_data

        return training_data, testing_data

    def append_flipped_team_and_outcomes_to_tournament(self, append_to_test=False):
        training_data = TournamentDataProcessor.append_flipped_team_and_outcomes(self.training_data)
        if append_to_test & len(self.testing_data) > 0:
            testing_data = TournamentDataProcessor.append_flipped_team_and_outcomes(self.testing_data)
        else:  #  If don't want to append to test data or cannot
            testing_data = self.testing_data
        return training_data, testing_data

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
                                  # 'region': self.tournament_region,
                                  'league': self.tournament_league,
                                  'team_id_1': team_ids[0], 'outcome_1': team_outcomes[0],
                                  'team_id_2': team_ids[1], 'outcome_2': team_outcomes[1]}, index=[0]))
        return pd.concat(game_tables, ignore_index=True)

    @staticmethod
    def swap_columns(df, cols1, cols2):
        """Swap the corresponding values of each of the columns of cols1 with cols2 and return a copy of the df"""
        if len(cols1) != len(cols2):
            raise ValueError("The number of columns to swap must be equal")
        df_copy = df.copy(deep=True)
        for i in range(len(cols1)):
            col_val = df_copy[cols1[i]].copy()
            df_copy[cols1[i]] = df_copy[cols2[i]]
            df_copy[cols2[i]] = col_val
        return df_copy

    @staticmethod
    def append_flipped_team_and_outcomes(data):
        # Want to ensure that there's symmetry between the teams and outcomes to prevent overfitting
        return pd.concat(
            [data, TournamentDataProcessor.swap_columns(data, ['team_id_1', 'outcome_1'], ['team_id_2', 'outcome_2'])],
            ignore_index=True)
