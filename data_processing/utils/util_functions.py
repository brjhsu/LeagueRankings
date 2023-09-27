def rename_features(df, features, prefix):
    return df.rename(columns=dict(zip(features, [f'{prefix}_{x}' for x in features])))

def get_last_game(df):
    return df[df['last_game']==True]