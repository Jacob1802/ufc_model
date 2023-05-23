from trueskill import rate_1vs1, Rating
import pandas as pd
import datetime as dt
import numpy as np
import time


def main():
    input_file = pd.read_csv("data/raw_fight_totals.csv")
    
    elo_df = set_elo(input_file)
    df = input_file.merge(elo_df)
    new = split_dataframe(df)
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
        'td_percent', 'kd', 'kd_received', 'sub_att', 'rev', 'ctrl', 'sig_reg_mixture', 'sig_reg_percent', 'elo_fighter', 'elo_change_fighter']
    
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
    df['elo_fighter'] = group['elo_fighter'].shift(1)
    df['elo_change_fighter'] = group['elo_change_fighter'].shift(1)

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
    temp_df['num_fights'] = grouped_fighters.cumcount()
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

    return df


def calculate_avg(df, column):
    df['avg_' + column + "_per_min"] = df[column] / df['total_fight_time']


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


def set_elo(df):
    # Initialize Elo ratings for each fighter
    elo_model_data = df[['fight_num', 'fighter_1', 'fighter_2', 'result_1']].copy()

    # Remove rows where fighter and opponent are the same
    elo_model_data = elo_model_data[elo_model_data['fighter_1'] != elo_model_data['fighter_2']]

    # Get first fight for each fight_pk
    elo_model_data = elo_model_data.groupby('fight_num').first().reset_index()

    # Calculate Elo ratings
    fighter_ratings = {}
    elo_table = []
    for i, row in elo_model_data.iterrows():
        fight_num = row['fight_num']
        fighter = row['fighter_1']
        opponent = row['fighter_2']
        res = row['result_1']

        if fighter not in fighter_ratings:
            fighter_ratings[fighter] = Rating()
        if opponent not in fighter_ratings:
            fighter_ratings[opponent] = Rating()
        if res == "W":
            fighter_ratings[fighter], fighter_ratings[opponent] = rate_1vs1(fighter_ratings[fighter], fighter_ratings[opponent])
        elif res == "L":
            fighter_ratings[opponent], fighter_ratings[fighter] = rate_1vs1(fighter_ratings[opponent], fighter_ratings[fighter])
            
        elo_table.append({'fighter_1': fighter,
                            'elo_fighter_1': fighter_ratings[fighter].mu,
                            'elo_change_fighter_1': fighter_ratings[fighter].mu - Rating().mu,
                            'fighter_2' : opponent,
                            'elo_fighter_2' : fighter_ratings[opponent].mu,
                            'elo_change_fighter_2' : fighter_ratings[opponent].mu - Rating().mu,
                            'fight_num': fight_num})

    # Combine Elo table with fighter table
    elo_table = pd.DataFrame(elo_table)
    
    return elo_table


if __name__ == "__main__":
    main()