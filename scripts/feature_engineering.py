from trueskill import rate_1vs1, Rating
import pandas as pd
import datetime as dt
import numpy as np
import time
import math

K = 32 # Maximum change in rating
INITIAL_RATING = 1200 # Starting rating for new fighters

# Dictionary to store each fighter's rating
fighter_ratings = {}

# get age

def main():
    input_file = pd.read_csv("data/raw_fight_totals.csv")
    
    # elo_df = set_elo(input_file)
    # return
    # df = input_file.merge(elo_df)
    new = split_dataframe(input_file)
    elo = set_elo(new)
    new[['elo', 'elo_change']] = elo[['elo', 'elo_change']]
    df = reformat_data(new)
    df = add_new_features(df)
    # Fill NaN values in columns with non-time data types
    dt_excluded_cols = df.select_dtypes(exclude=['timedelta64[ns]'])
    df[dt_excluded_cols.columns] = dt_excluded_cols.fillna(0)

    # Fill NaT values in columns with time data types
    dt_cols = df.select_dtypes(include=['timedelta64[ns]'])
    df[dt_cols.columns] = dt_cols.fillna(pd.Timedelta(0))
    
    df.to_csv("data/fights.csv", index=False)


def split_dataframe(df):
    details_columns = ['fight_num', 'date', 'fighter', 'result', 'weight_class']
    method_columns = ['method', 'time format', 'time', 'round']
    cols = ['sig_str_landed', 'sig_str_attempts', 'sig_str_received', 'sig_str_avoided', 'sig_str_percent', 'str_landed', 
        'str_attempts', 'str_received', 'str_avoided', 'str_percent', 'td_landed', 'td_attempts', 'td_received', 'td_avoided', 
        'td_percent', 'kd', 'kd_received', 'sub_att', 'rev', 'ctrl', 'sig_reg_mixture', 'sig_reg_percent']
    #'elo', 'elo_change'
    
    new_columns = details_columns + method_columns + cols

    new_data = []
    
    for _, row in df.iterrows():
        date_row = [row['fight_num'], row['date']]
        result_1_row = [row['result_1']]
        result_2_row = [row['result_2']]
        fighter_1_row = [f"{row['fighter_1']}"]
        fighter_2_row = [f"{row['fighter_2']}"]
        weightclass_row = [row['weight_class']]
        method_row = [row[col] for col in method_columns]
        
        row_1 = [row[f"{col}_1"] for col in cols]
        row_2 = [row[f"{col}_2"] for col in cols]
        
        new_data.append(date_row + fighter_1_row + result_1_row + weightclass_row + method_row + row_1)
        new_data.append(date_row + fighter_2_row + result_2_row + weightclass_row + method_row + row_2)

    new_df = pd.DataFrame(new_data, columns=new_columns)
    return new_df


def reformat_data(df):
    ufc_weight_classes = [
        "men_strawweight",
        "men_flyweight",
        "men_bantamweight",
        "men_featherweight",
        "men_lightweight",
        "men_welterweight",
        "men_middleweight",
        "men_light_heavyweight",
        "men_heavyweight",
        "womens_strawweight",
        "womens_flyweight",
        "womens_bantamweight",
        "womens_featherweight",
    ]

    df['method'] = df['method'].replace({
        "TKO - Doctor's Stoppage": "KO/TKO",
        "Decision - Unanimous": "Decision",
        "Decision - Split": "Decision",
        "Decision - Majority": "Decision",
        "Overturned": "Other",
        "DQ": "Other",
        "Could Not Continue": "Other"
    })
    group = df.groupby('fighter')
    # df['elo'] = group['elo'].shift(1)
    # df['elo_change'] = group['elo_change'].shift(1)

    for col in ["sig_str_percent", "td_percent"]:
        df[col] = df[col].str.strip("%").replace("---", "0").fillna(0).apply(lambda x: int(x) / 100)
    df['date'] = pd.to_datetime(df['date']).fillna(pd.Timedelta(0))
    df['weight_class'] = df['weight_class'].str.replace("UFC", "").str.replace("Title", "")
    df['weight_class'] = df['weight_class'].str.replace("Bout", "").str.strip().str.replace(" ", "_")
    df['weight_class'] = df['weight_class'].str.replace("Women's", "womens").str.lower()
    df['weight_class'] = df['weight_class'].apply(lambda x: "men_" + x if not x.startswith("women") else x)
    df['weight_class'] = df['weight_class'].apply(lambda x: "catchweight" if x.lower() not in ufc_weight_classes else x)
    # Remove any 'OT' occurrences if present and convert to datetime object
    df['time'] = pd.to_datetime(df['time'].str.replace('OT', ''), format='%M:%S').apply(convert_to_minutes).fillna(pd.Timedelta(0)) # Convert to minutes  
    
    df['ctrl'] = pd.to_datetime(df['ctrl'], format='%H:%M:%S', errors='coerce').fillna(pd.to_datetime('00:00:00', format='%H:%M:%S')).apply(convert_to_minutes).fillna(pd.Timedelta(0)) # Convert to minutes
    
    df = df.rename(columns={'round': 'round_finished'})
    # df['rounds'] = df['time format'].apply(lambda x: "1" if x == "No Time Limit" else x).str.extract('^(\d+)', expand=False).astype(int)
    df = df.drop('time format', axis=1)
    return df


def add_new_features(df):
    totals = ['kd', 'kd_received','round_finished','sig_str_landed','sig_str_received','sig_str_attempts',
        'sig_str_avoided','str_received','str_attempts', 'str_avoided','str_landed',
        'td_received','td_attempts','td_avoided','td_landed']
        
    damage_cols = ['sig_str_landed',
       'sig_str_attempts', 'sig_str_received', 'sig_str_avoided',
       'str_landed', 'str_attempts', 'str_received',
       'str_avoided']

    differentials = ['sig_str_landed', 'str_landed']
    
    lags = damage_cols + ['kd', 'kd_received',
       'td_landed', 'td_attempts', 'td_received', 'td_avoided',
       'sub_att', 'rev']
    
    accuracy = ['sig_str', 'str', 'td']

    methods = ['Decision','KO/TKO','Other','Submission']
    temp_df = pd.DataFrame()
    grouped_fighters = df.groupby('fighter', group_keys=False)
    
    temp_df['fighter'] = df['fighter']
    temp_df['fight_num'] = df['fight_num']
    temp_df['num_fights'] = grouped_fighters.cumcount().shift(1)
    df['total_fight_time'] = grouped_fighters['time'].cumsum()
    temp_df['days_since_last_fight'] = grouped_fighters['date'].apply(lambda x: x - x.shift(1)).fillna(pd.Timedelta(0))
    
    # Total wins, losses & w/l by decison, ko, sub, other
    for result in ["W", "L"]:
        temp_df["total_" + result.lower()] = grouped_fighters.apply(lambda x: (x['result'] == result).shift(1).cumsum())
        for method in methods:
            temp_df[result.lower() + "_" + method.lower()] = grouped_fighters.apply(lambda x: ((x['result'] == result) & (x['method'] == method)).shift(1).cumsum())

    for col in accuracy:
        temp_df[col + '_accuracy'] = df[col + '_attempts'] / df[col + '_landed']

    
    for col in differentials:
        differential(df, col)
        
    for col in lags + ['sig_str_landed_differential', 'str_landed_differential']:
        for i in range(1, 4):
            lag(df, col, i)
        
        calculate_weighted_avg(df, col)
        calculate_avg(df, col)

    for col in totals:
        if col == "round_finished":
            df["foo"] = grouped_fighters[col].cumsum()
            temp_df['total_rounds'] = grouped_fighters['foo'].shift(1)
        else:
            df["foo1"] = grouped_fighters[col].cumsum()
            temp_df["total_" + col] = grouped_fighters['foo1'].shift(1)
            
    df = df.drop(['foo', 'foo1'], axis=1)

    # Iterate over each group
    for _, group in grouped_fighters:
        # Reset streak counters for each group
        current_win_streak = 0
        current_loss_streak = 0
        
        # Iterate over rows in the group
        for index, row in group.iterrows():
            # Update streak columns for the current row
            temp_df.at[index, 'win_streak'] = current_win_streak
            temp_df.at[index, 'loss_streak'] = current_loss_streak
            if row['result'] == 'W':
                # Increment win streak and reset loss streak
                current_win_streak += 1
                current_loss_streak = 0
            elif row['result'] == 'L':
                # Increment loss streak and reset win streak
                current_loss_streak += 1
                current_win_streak = 0
            else:
                current_win_streak = 0
                current_loss_streak = 0
                
    df = df.merge(temp_df, on=['fighter', 'fight_num'])
    df['total_fight_time'] = grouped_fighters['total_fight_time'].shift(1)
    return df


def calculate_avg(df, column):
    df['avg_' + column + "_per_min"] = df[column] / df['total_fight_time']
    df['avg_' + column + "_per_min"] = df.groupby('fighter')['avg_' + column + "_per_min"].shift(1)
    # df['avg_' + column + "_per_min"] = df.groupby('fighter')[column] / df.groupby('fighter')['total_fight_time']


def differential(df, column):
    group = df.groupby('fight_num',group_keys=False)[column]
    # Get difference of stats
    differ = group.diff()
    nan_indices = differ.isna()
    # Back fill nan with counterpart
    differ = differ.fillna(method='bfill')
    # Inverse counter part with nan indices
    differ[nan_indices] *= -1
    
    df[column + "_differential"] = differ.astype(float)


def lag(df, column, step):
    df[column + '_lag' + str(step)] = df.groupby('fighter')[column].shift(step).fillna(0).astype(int)


def calculate_weighted_avg(df, column):
    group = df.groupby('fighter')[column]

    # Calculate weighted avg of previous 3 fights
    three = 0.6 * group.shift(1) + 0.3 * group.shift(2) + 0.1 * group.shift(3)
    # Calculate weighted avg of previous 2 fights if only 2 fights
    two = three.fillna(0.7 * group.shift(1) + 0.3 * group.shift(2))
    # Fill if only 1 fight
    one = two.fillna(group.shift(1))
    weighted_avg = one.fillna(0)
    
    df[column + "_weighted_avg"] = weighted_avg.astype(float)


def convert_to_minutes(x):
    time_delta = pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
    return time_delta.total_seconds() / 60


def calculate_expected_win_probability(rating1, rating2):
    """ Calculate the expected win probability for the first fighter based on the Elo rating of both fighters. """
    return 1 / (1 + math.pow(10, (rating2 - rating1) / 400))


def update_ratings(winner, loser):
    """ Update the ratings of the winner and loser based on the outcome of their fight. """
    winner_rating = fighter_ratings.get(winner, INITIAL_RATING)
    loser_rating = fighter_ratings.get(loser, INITIAL_RATING)
    expected_win_probability = calculate_expected_win_probability(winner_rating, loser_rating)
    winner_new_rating = winner_rating + K * (1 - expected_win_probability)
    loser_new_rating = loser_rating + K * (0 - expected_win_probability)
    return winner_new_rating, loser_new_rating


def set_elo(df):
    rows = []
    group = df.groupby("fight_num")
    for group_i, group_df in group:
        other = False
        row_1 = {'fight_num' : group_i, 'fighter' : group_df.iloc[0]['fighter']}
        row_2 = {'fight_num' : group_i, 'fighter' : group_df.iloc[1]['fighter']}
        if group_df.iloc[0]['result'] == "W":
            winner = group_df.iloc[0]['fighter']
            loser = group_df.iloc[1]['fighter']
        elif group_df.iloc[1]['result'] == "W":
            winner = group_df.iloc[1]['fighter']
            loser = group_df.iloc[0]['fighter']
        else:
            other = True
            row_1['elo'] = fighter_ratings.get(group_df.iloc[0]['fighter'])
            row_1['elo_change'] = 0
            row_2['elo'] = fighter_ratings.get(group_df.iloc[1]['fighter'])
            row_2['elo_change'] = 0
            
        if not other:
            winner_rating = fighter_ratings.get(winner, 1200)
            loser_rating = fighter_ratings.get(loser, 1200)
            winner_new_rating, loser_new_rating = update_ratings(winner, loser)
            fighter_ratings[winner] = winner_new_rating
            fighter_ratings[loser] = loser_new_rating
            if row_1['fighter'] == winner:
                row_1['elo'] = winner_new_rating
                row_1['elo_change'] = winner_new_rating - winner_rating
                row_2['elo'] = loser_new_rating
                row_2['elo_change'] = loser_new_rating - loser_rating
            elif row_2['fighter'] == winner:
                row_1['elo'] = winner_new_rating
                row_1['elo_change'] = winner_new_rating - winner_rating
                row_2['elo'] = loser_new_rating
                row_2['elo_change'] = loser_new_rating - loser_rating
        
        rows.append(row_1)
        rows.append(row_2)
    
    elo_table = pd.DataFrame(rows)
    return elo_table


if __name__ == "__main__":
    main()