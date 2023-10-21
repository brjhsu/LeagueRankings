from difflib import SequenceMatcher

def rename_features(df, features, prefix):
    return df.rename(columns=dict(zip(features, [f'{prefix}_{x}' for x in features])))

def get_last_game(df):
    return df[df['last_game']==True]

def find_closest_key(string, dictionary):
    closest_key = None
    closest_distance = 0
    
    for key in dictionary.keys():
        distance = SequenceMatcher(None, string, key).ratio()
        if distance > closest_distance:
            closest_key = key
            closest_distance = distance
    print(closest_key)
    return dictionary[closest_key]