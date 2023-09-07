from datetime import datetime
import pandas as pd
import numpy as np


class GameFeaturesGenerator:
    TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

    @staticmethod
    def get_event_time(start_time, event_time):
        return (datetime.strptime(event_time, GameFeaturesGenerator.TIME_FORMAT) - start_time).total_seconds()

    @staticmethod
    def flip_team_id(team_id):
        """
        Flips team_id from 100 to 200 and vice versa. Needed for when processing turret fall events because we want to update the feature for the team that lost the turret
        """
        if team_id == 100:
            return 200
        elif team_id == 200:
            return 100
        else:
            print("Error: team_id is not 100 or 200")

    def __init__(self, game_data, mapping_data):
        self.game_data = game_data
        self.game_start_time = datetime.strptime(game_data[0]['eventTime'], GameFeaturesGenerator.TIME_FORMAT)
        self.team_id_mapping = {"100": mapping_data['teamMapping']['100'], "200": mapping_data['teamMapping']['200']}
        self.esports_game_id = mapping_data['esportsGameId']  # Looks like '110310652412257228'
        self.platform_game_id = mapping_data['platformGameId']  # Looks lik {'esportsGameId': '110310652412257228'
        # Create flags for first events (e.g. first turret kill, first dragon kill, etc.). These can only happen once (for both teams)
        self.first_herald_flag = True
        self.first_dragon_flag = True
        self.first_baron_flag = True
        self.first_turret_flag = True
        self.first_inhibitor_flag = True
        self.first_kill_flag = True

        # Create features for each team
        # EPIC MONSTER EVENT FEATURES
        self.IMPORTANT_MONSTER_TYPES = ["riftHerald", "dragon", "baron"]
        epic_monster_kill_event_features = [
            ["first_" + monster_type + "_ind", "first_" + monster_type + "_time", "num_" + monster_type] for
            monster_type
            in self.IMPORTANT_MONSTER_TYPES]
        epic_monster_kill_event_features = [item for sublist in epic_monster_kill_event_features for item in
                                            sublist]  # Flatten list of lists

        # BUILDING DESTROYED EVENT FEATURES
        self.IMPORTANT_BUILDING_TYPES = ["turret", "inhibitor"]
        building_destroyed_event_features = [
            ["first_" + building_type + "_ind", "first_" + building_type + "_time", "num_" + building_type] for
            building_type
            in self.IMPORTANT_BUILDING_TYPES]
        building_destroyed_event_features = [item for sublist in building_destroyed_event_features for item in
                                             sublist]  # Flatten list of lists

        # CHAMPION KILL EVENT FEATURES
        champion_kill_event_features = ["first_kill_ind", "first_kill_time", "num_kills"]

        # GAME METADATA FEATURES
        # TODO: Also need a feature for whether this was a domestic or international game (e.g. LCS vs. Worlds)
        game_metadata_features = ["game_end_time"]

        all_features = (["platformGameId", "esportsGameId", "team_id", "start_time", "outcome"] +
                        epic_monster_kill_event_features + building_destroyed_event_features + champion_kill_event_features + game_metadata_features)
        self.team_features = {"100": {feature: np.nan for feature in all_features},
                              "200": {feature: np.nan for feature in all_features}}
        # Set all "num_" features to 0
        for team_id in ["100", "200"]:
            for feature in all_features:
                if feature.startswith("num_"):
                    self.team_features[team_id][feature] = 0

    def process_epic_monster_kill_event(self, event, team_features):
        monster_type = event['monsterType']
        if monster_type in self.IMPORTANT_MONSTER_TYPES:
            event_time = event['eventTime']
            time_of_kill = GameFeaturesGenerator.get_event_time(self.game_start_time, event_time)
            killer_team_id = str(event['killerTeamID'])
            victim_team_id = str(GameFeaturesGenerator.flip_team_id(event['killerTeamID']))
            if self.first_herald_flag:
                team_features[killer_team_id]['first_riftHerald_ind'] = 1
                team_features[victim_team_id]['first_riftHerald_ind'] = 0
                team_features[killer_team_id]['first_riftHerald_time'] = time_of_kill
                self.first_herald_flag = False
            if self.first_dragon_flag:
                team_features[killer_team_id]['first_dragon_ind'] = 1
                team_features[victim_team_id]['first_dragon_ind'] = 0
                team_features[killer_team_id]['first_dragon_time'] = time_of_kill
                self.first_dragon_flag = False
            if self.first_baron_flag:
                team_features[killer_team_id]['first_baron_ind'] = 1
                team_features[victim_team_id]['first_baron_ind'] = 0
                team_features[killer_team_id]['first_baron_time'] = time_of_kill
                self.first_baron_flag = False
            team_features[killer_team_id]['num_' + monster_type] += 1

    def process_building_destroyed_event(self, event, team_features):
        building_type = event['buildingType']
        if building_type in self.IMPORTANT_BUILDING_TYPES:
            event_time = event['eventTime']
            time_of_event = GameFeaturesGenerator.get_event_time(self.game_start_time, event_time)
            team_id = str(GameFeaturesGenerator.flip_team_id(event['teamID']))
            victim_team_id = str(event['teamID'])
            if self.first_turret_flag:
                team_features[team_id]['first_turret_ind'] = 1
                team_features[victim_team_id]['first_turret_ind'] = 0
                team_features[team_id]['first_turret_time'] = time_of_event
                self.first_turret_flag = False
            if self.first_inhibitor_flag:
                team_features[team_id]['first_inhibitor_ind'] = 1
                team_features[victim_team_id]['first_inhibitor_ind'] = 0
                team_features[team_id]['first_inhibitor_time'] = time_of_event
                self.first_inhibitor_flag = False
            team_features[team_id]['num_' + building_type] += 1

    def process_champion_kill_event(self, event, team_features):
        event_time = event['eventTime']
        time_of_event = GameFeaturesGenerator.get_event_time(self.game_start_time, event_time)
        killer_team_id = str(event['killerTeamID'])
        victim_team_id = str(GameFeaturesGenerator.flip_team_id(event['killerTeamID']))
        if self.first_kill_flag:
            team_features[killer_team_id]['first_kill_ind'] = 1
            team_features[victim_team_id]['first_kill_ind'] = 0
            team_features[killer_team_id]['first_kill_time'] = time_of_event
            self.first_kill_flag = False
        team_features[killer_team_id]['num_kills'] += 1

    def process_game_end_event(self, event, team_features):
        # We only call this method for the final event, so if for some reason there is no game_end event then discard the data
        if event['eventType'] == "game_end":
            event_time = event['eventTime']
            time_of_event = GameFeaturesGenerator.get_event_time(self.game_start_time, event_time)
            team_features['100']['game_end_time'] = time_of_event
            team_features['200']['game_end_time'] = time_of_event
            winning_team = event['winningTeam']
            losing_team = GameFeaturesGenerator.flip_team_id(winning_team)
            team_features[str(winning_team)]['outcome'] = 1
            team_features[str(losing_team)]['outcome'] = 0
        else:
            raise Exception("Error: event is not a game_end event")

    def process_game(self):
        """
        Loops through all events in the game and updates the team features as they occur
        :return: a pandas dataframe with the features for each team and label "outcome"
        """
        for event in self.game_data:
            if event['eventType'] == "epic_monster_kill":
                self.process_epic_monster_kill_event(event, self.team_features)
            elif event['eventType'] == "building_destroyed":
                self.process_building_destroyed_event(event, self.team_features)
            elif event['eventType'] == "champion_kill":
                self.process_champion_kill_event(event, self.team_features)
            elif event['eventType'] == "game_end":
                self.process_game_end_event(event, self.team_features)
            else:
                continue

        # Now we have all the datapoints, we can create a dataframe with the team ID, label, start_time, and features
        # First assign the rest of the metadata
        rows = []
        for team_id in ["100", "200"]:
            # Assign metadata (same for both teams)
            self.team_features[team_id]['platformGameId'] = self.platform_game_id
            self.team_features[team_id]['esportsGameId'] = self.esports_game_id
            self.team_features[team_id]['start_time'] = self.game_start_time

            # Assing team-specific data
            self.team_features[team_id]['team_id'] = self.team_id_mapping[team_id]
            rows.append(pd.DataFrame.from_dict(self.team_features[team_id], orient='index').transpose())
        return pd.concat(rows, ignore_index=True)
