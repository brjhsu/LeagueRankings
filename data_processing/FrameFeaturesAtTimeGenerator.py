import pandas as pd

class FrameFeaturesAtTimeGenerator:
    def __init__(self, event, event_time):
        self.event = event
        self.event_time = event_time  # Must be time in SECONDS, not minutes

        self.team_features_from_event_data = ['inhibKills', 'towerKills', 'baronKills', 'dragonKills', 'assists',
                                              'championsKills', 'totalGold', 'deaths']
        self.team_features_from_frame_stats = ['MINIONS_KILLED', 'NEUTRAL_MINIONS_KILLED', 'VISION_SCORE']
        self.participant_mapping = {
            '1': 'top',
            '2': 'jungle',
            '3': 'mid',
            '4': 'bot',
            '5': 'support',
            '6': 'top',
            '7': 'jungle',
            '8': 'mid',
            '9': 'bot',
            '10': 'support'
        }

    def extract_participant_stats(self, participant_data):
        """
        Extracts basic stats from participant data
        """
        participant_id = str(participant_data['participantID'])
        team_id = str(participant_data['teamID'])
        participant_stats_basic = {'XP': participant_data['XP'], 'totalGold': participant_data['totalGold'],
                                   'currentGold': participant_data['currentGold']}
        participant_stats_frames = participant_data['stats']
        participant_stats_granular = self.extract_participants_granular_stats(participant_stats_frames)

        # Combine the basic and granular stats
        participant_stats = {'participant_id': participant_id, 'team_id': team_id, **participant_stats_basic,
                             **participant_stats_granular}
        return participant_stats

    def extract_participants_granular_stats(self, stats_frames):
        """
        Extracts granular stats from participant data (e.g. minions killed, jungle minions killed, KDA, etc.)
        :param stats_frame: the 'stats' item from the 'status_update' event
        :return: a dictionary of granular stats
        """
        stat_record = {}
        stats_to_record = {'MINIONS_KILLED', 'NEUTRAL_MINIONS_KILLED', 'NEUTRAL_MINIONS_KILLED_YOUR_JUNGLE',
                           'NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE', 'CHAMPIONS_KILLED', 'NUM_DEATHS', 'ASSISTS',
                           'WARD_PLACED', 'WARD_KILLED', 'VISION_SCORE', 'TOTAL_DAMAGE_DEALT_TO_CHAMPIONS',
                           'TOTAL_DAMAGE_TAKEN', 'TOTAL_DAMAGE_DEALT_TO_BUILDINGS', 'TOTAL_DAMAGE_DEALT_TO_OBJECTIVES'}
        # other stats to consider
        # ['TOTAL_TIME_CROWD_CONTROL_DEALT_TO_CHAMPIONS', 'TOTAL_HEAL_ON_TEAMMATES', 'TOTAL_DAMAGE_SHIELDED_ON_TEAMMATES']
        for stat in stats_frames:
            stat_name = stat['name']
            if stat_name in stats_to_record:
                stat_record[stat_name] = stat['value']
        return stat_record

    # Now need to convert frame_stats into a single row of data for each team (100 and 200)
    def create_participant_features(self, frame_stats):
        """
        Creates higher-order features for each participant (e.g. KDA, minions killed, etc.)
        :param frame_stats: a dataframe with stats for each participant for a single team at a single point in time
        :return: a dataframe with features for each participant
        """
        # Create features for each participant
        # For KDA, we need to add 0.1 to the denominator to avoid dividing by 0
        num_minutes = frame_stats['event_time'] / 60
        frame_stats['KDA'] = (frame_stats['CHAMPIONS_KILLED'] + frame_stats['ASSISTS']) / (
                    frame_stats['NUM_DEATHS'] + 1.0)
        frame_stats['cs_per_min'] = (frame_stats['MINIONS_KILLED'] + frame_stats[
            'NEUTRAL_MINIONS_KILLED']) / num_minutes
        frame_stats['xp_per_min'] = frame_stats['XP'] / num_minutes
        frame_stats['gold_per_min'] = frame_stats['totalGold'] / num_minutes
        frame_stats['vision_per_min'] = frame_stats['VISION_SCORE'] / num_minutes
        frame_stats['wards_per_min'] = (frame_stats['WARD_PLACED'] + frame_stats['WARD_KILLED']) / num_minutes
        frame_stats['trade_efficiency'] = frame_stats['TOTAL_DAMAGE_DEALT_TO_CHAMPIONS'] / (
                    frame_stats['TOTAL_DAMAGE_TAKEN'] + 1.0)  # Higher for poke-heavy champions
        frame_stats['damage_to_champions_per_min'] = frame_stats['TOTAL_DAMAGE_DEALT_TO_CHAMPIONS'] / num_minutes
        frame_stats['damage_to_buildings_per_min'] = frame_stats['TOTAL_DAMAGE_DEALT_TO_BUILDINGS'] / num_minutes
        frame_stats['gold_spent'] = frame_stats['totalGold'] - frame_stats['currentGold']
        frame_stats['share_of_minions_stolen'] = frame_stats['NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE'] / (
                    frame_stats['NEUTRAL_MINIONS_KILLED'] + 1.0)
        frame_stats['share_of_team_gold'] = frame_stats['totalGold'] / frame_stats['totalGold'].sum()
        frame_stats['share_of_team_xp'] = frame_stats['XP'] / frame_stats['XP'].sum()
        frame_stats['share_of_team_damage_to_champions'] = frame_stats['TOTAL_DAMAGE_DEALT_TO_CHAMPIONS'] / frame_stats[
            'TOTAL_DAMAGE_DEALT_TO_CHAMPIONS'].sum()
        frame_stats['share_of_team_damage_to_buildings'] = frame_stats['TOTAL_DAMAGE_DEALT_TO_BUILDINGS'] / frame_stats[
            'TOTAL_DAMAGE_DEALT_TO_BUILDINGS'].sum()
        frame_stats['share_of_team_damage_to_objectives'] = frame_stats['TOTAL_DAMAGE_DEALT_TO_OBJECTIVES'] / \
                                                            frame_stats['TOTAL_DAMAGE_DEALT_TO_OBJECTIVES'].sum()
        return frame_stats

    def create_team_features(self, frame_stats, event_data):
        """
        Creates team-level features (e.g. total gold, total share of dragons/barons taken, etc.)
        :param frame_stats: A previously generated dataframe with features for each participant
        :param event_data: The event data for the current event
        :return:
        """
        # Extract the team_features for each team based on event_data['teams'][i][feature]
        # Then compute the share of each feature for each team (whenever both features are zero, set the share to 0.5)
        # Then add the share of each feature to the frame_stats dataframe

        # First extract the team features from event_data
        team_features = {'100': {}, '200': {}}
        for team in event_data['teams']:
            team_id = str(team['teamID'])
            for feature in self.team_features_from_event_data:
                team_features[team_id][feature] = team[feature]

        # Compute the data for each team as the sum of the data for each participant on each team
        for team in team_features:
            for feature in self.team_features_from_frame_stats:
                team_features[team][feature] = frame_stats[frame_stats['team_id'] == team][feature].sum()

        # Now compute the share of each feature for each team, setting the share to 0.5 if they are both 0
        for team in team_features:
            for feature in self.team_features_from_event_data + self.team_features_from_frame_stats:
                total_value = (team_features['100'][feature] + team_features['200'][feature])
                if total_value == 0:
                    team_features[team]['team_share_of_' + feature] = 0.5
                else:
                    team_features[team]['team_share_of_' + feature] = team_features[team][feature] / total_value

        # Drop all features that are not 'share_of_' features
        for team in team_features:
            for feature in self.team_features_from_event_data + self.team_features_from_frame_stats:
                del team_features[team][feature]
        return team_features

    def create_diff_features(self, frame_stats):
        # Create features based on the difference between the player on the row and the player on the other team (row +5)
        # Do this for the first 5 players
        # For diff metrics, the value for the other team is the negative of the values for the first team
        # For share metrics, the value for the other team is 1- the value for the first team

        # DIFF METRICS
        # Overall relative performance indicators
        frame_stats['KDA_diff'] = (frame_stats['KDA'] - frame_stats['KDA'].shift(-5))
        frame_stats['cs_per_min_diff'] = (frame_stats['cs_per_min'] - frame_stats['cs_per_min'].shift(-5))
        frame_stats['xp_per_min_diff'] = (frame_stats['xp_per_min'] - frame_stats['xp_per_min'].shift(-5))
        frame_stats['gold_per_min_diff'] = (frame_stats['gold_per_min'] - frame_stats['gold_per_min'].shift(-5))
        frame_stats['vision_per_min_diff'] = (frame_stats['vision_per_min'] - frame_stats['vision_per_min'].shift(-5))
        frame_stats['wards_per_min_diff'] = (frame_stats['wards_per_min'] - frame_stats['wards_per_min'].shift(-5))

        # Lane trade indicators
        frame_stats['trade_efficiency_diff'] = (
                    frame_stats['trade_efficiency'] - frame_stats['trade_efficiency'].shift(-5))
        frame_stats['damage_to_champions_per_min_diff'] = (
                    frame_stats['damage_to_champions_per_min'] - frame_stats['damage_to_champions_per_min'].shift(-5))
        frame_stats['damage_to_buildings_per_min_diff'] = (
                    frame_stats['damage_to_buildings_per_min'] - frame_stats['damage_to_buildings_per_min'].shift(-5))

        # SHARE METRICS
        # Opponent dominance indicators
        frame_stats['lane_cs_dominance'] = (
                    frame_stats['cs_per_min'] / (frame_stats['cs_per_min'] + frame_stats['cs_per_min'].shift(-5)))
        frame_stats['lane_damage_to_champions_dominance'] = (frame_stats['damage_to_champions_per_min'] /
                                                             (1.0 + frame_stats['damage_to_champions_per_min'] +
                                                              frame_stats['damage_to_champions_per_min'].shift(-5)))
        frame_stats['lane_damage_to_buildings_dominance'] = (frame_stats['damage_to_buildings_per_min'] /
                                                             (1.0 + frame_stats['damage_to_buildings_per_min'] +
                                                              frame_stats['damage_to_buildings_per_min'].shift(-5)))

        # AGGREGATE METRICS
        # Now get the values for the other team (row +5) by either doing negative or 1- the values we just computed. But ONLY operate on rows 5-9
        # DIFF METRICS
        frame_stats.loc[5:10, 'KDA_diff'] = -frame_stats.loc[0:4, 'KDA_diff'].values
        frame_stats.loc[5:10, 'cs_per_min_diff'] = -frame_stats.loc[0:4, 'cs_per_min_diff'].values
        frame_stats.loc[5:10, 'xp_per_min_diff'] = -frame_stats.loc[0:4, 'xp_per_min_diff'].values
        frame_stats.loc[5:10, 'gold_per_min_diff'] = -frame_stats.loc[0:4, 'gold_per_min_diff'].values
        frame_stats.loc[5:10, 'vision_per_min_diff'] = -frame_stats.loc[0:4, 'vision_per_min_diff'].values
        frame_stats.loc[5:10, 'trade_efficiency_diff'] = -frame_stats.loc[0:4, 'trade_efficiency_diff'].values
        frame_stats.loc[5:10, 'damage_to_champions_per_min_diff'] = -frame_stats.loc[0:4,
                                                                     'damage_to_champions_per_min_diff'].values
        frame_stats.loc[5:10, 'damage_to_buildings_per_min_diff'] = -frame_stats.loc[0:4,
                                                                     'damage_to_buildings_per_min_diff'].values
        frame_stats.loc[5:10, 'wards_per_min_diff'] = -frame_stats.loc[0:4, 'wards_per_min_diff'].values
        # SHARE METRICS
        frame_stats.loc[5:10, 'lane_cs_dominance'] = 1 - frame_stats.loc[0:4, 'lane_cs_dominance'].values
        frame_stats.loc[5:10, 'lane_damage_to_champions_dominance'] = 1 - frame_stats.loc[0:4,
                                                                          'lane_damage_to_champions_dominance'].values
        frame_stats.loc[5:10, 'lane_damage_to_buildings_dominance'] = 1 - frame_stats.loc[0:4,
                                                                          'lane_damage_to_buildings_dominance'].values

        return frame_stats

    def melt_frame_stats(self, frame_stats):
        """
        Melts the frame_stats dataframe so that each row is a single feature for all participants
        """
        # Rename the participant_id such that it maps to ["top","jungle","mid","bot","support"] for each team
        frame_stats['participant_id'] = frame_stats['participant_id'].map(self.participant_mapping)
        frame_stats = frame_stats.melt(id_vars=['participant_id'], value_vars=frame_stats.columns)
        frame_stats['name'] = frame_stats['participant_id'] + '_' + frame_stats['variable']
        frame_stats = frame_stats[['name', 'value']].set_index('name').transpose()
        return frame_stats

    def process_frame(self, time=None):
        individual_stats = []
        for participant_data in self.event['participants']:
            individual_stats.append(self.extract_participant_stats(participant_data))

        # Convert the list of dictionaries into a single dataframe and ensure that it's sorted by partipant_id when evaluated as a number
        frame_stats = pd.DataFrame.from_dict(individual_stats).sort_values(by=['participant_id'],
                                                                           key=lambda x: x.astype(int))

        # Add the time column
        frame_stats['event_time'] = self.event_time

        frame_stats_100 = frame_stats[frame_stats['team_id'] == '100']
        frame_stats_100 = self.create_participant_features(frame_stats[frame_stats['team_id'] == '100'])
        frame_stats_200 = frame_stats[frame_stats['team_id'] == '200']
        frame_stats_200 = self.create_participant_features(frame_stats[frame_stats['team_id'] == '200'])

        frame_stats = pd.concat([frame_stats_100, frame_stats_200])
        # Get the team features
        team_features_stats = self.create_team_features(frame_stats, self.event)
        # Get the lane difference features
        frame_stats = self.create_diff_features(frame_stats)

        # Lastly turn these into per-player features by splitting again by team_id, pivoting on participant_id, and then joining back together
        frame_stats_100 = frame_stats[frame_stats['team_id'] == '100'].drop(['team_id', 'event_time'], axis=1)
        frame_stats_200 = frame_stats[frame_stats['team_id'] == '200'].drop(['team_id', 'event_time'], axis=1)

        # Pivot and convert to dictionary format
        frame_stats_100 = self.melt_frame_stats(frame_stats_100)
        frame_stats_100 = {x: y for x, y in zip(frame_stats_100.columns.values, frame_stats_100.values[0])}
        all_stats_100 = {**team_features_stats['100'], **frame_stats_100}
        if time is None:
            time = round(self.event_time / 60)
        all_stats_100 = {x + f'_at_{time}': y for x, y in all_stats_100.items()}

        frame_stats_200 = self.melt_frame_stats(frame_stats_200)
        frame_stats_200 = {x: y for x, y in zip(frame_stats_200.columns.values, frame_stats_200.values[0])}
        all_stats_200 = {**team_features_stats['200'], **frame_stats_200}
        all_stats_200 = {x + f'_at_{time}': y for x, y in all_stats_200.items()}

        return all_stats_100, all_stats_200