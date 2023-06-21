from sklearn.preprocessing import LabelEncoder
from future_fights import get_future_matchups
import pandas as pd
import math
import re

INITIAL_RATING = 1200 # Starting rating for new fighters

# Dictionary to store each fighter's rating
fighter_ratings = {}


def main():
    input_file = pd.read_csv("data/raw_fight_totals.csv")
    new = split_dataframe(input_file)
    
    elo = set_elo(new)
    new[['curr_elo', 'curr_elo_change', 'future_elo', 'future_elo_change']] = elo[['curr_elo', 'curr_elo_change', 'future_elo', 'future_elo_change']]
    df = reformat_data(new)
    predictions_df = add_new_features(df, 0, True)
    predictions_csv(predictions_df)
    df = add_new_features(df, 1)
    # Fill NaN values in columns with non-time data types
    dt_excluded_cols = df.select_dtypes(exclude=['timedelta64[ns]'])
    df[dt_excluded_cols.columns] = dt_excluded_cols.fillna(0)

    # Fill NaT values in columns with time data types
    dt_cols = df.select_dtypes(include=['timedelta64[ns]'])
    df[dt_cols.columns] = dt_cols.fillna(pd.Timedelta(0))
    
    df['days_since_last_fight'] = df['days_since_last_fight'].astype(str)
    df['days_since_last_fight'] = pd.to_numeric(df['days_since_last_fight'].str.replace(' days', ''))

    joined_df = join_rows(df)
    joined_df.to_csv("data/fights.csv", index=False)


def predictions_csv(df):
    """
    Process the predictions dataframe and save the modified version as a CSV file.

    Args:
        predictions_df (pandas.DataFrame): The dataframe containing predictions data.

    Returns:
        None
    """

    # List of columns to exclude from the final dataframe
    exclude = [
        'result', 'result_code', 'weight_class', 'method', 'time', 'round_finished', 'sig_str_landed',
        'sig_str_attempts', 'sig_str_received', 'sig_str_avoided', 'sig_str_percent', 'str_landed',
        'str_attempts', 'str_received', 'str_avoided', 'str_percent', 'td_landed', 'td_attempts',
        'td_received', 'td_avoided', 'td_percent', 'kd', 'kd_received', 'sub_att', 'rev', 'ctrl',
        'sig_reg_mixture', 'sig_reg_percent'
    ]

    # Get future matchups, date, and weight classes
    date, matchups, weightclasses = get_future_matchups()
    # Flatten matchups into a list of fighters
    fighters = [fighter for matchup in matchups for fighter in matchup]
    # Convert date to a pandas Timestamp object
    date = pd.Timestamp(date)
    # Filter dataframe to include only fighters in the matchups
    predictions_df = df[df['fighter'].isin(fighters)].copy()
    # Process each matchup
    for i, matchup in enumerate(matchups):
        try:
            # Get the index of the last occurrence of each fighter in the predictions dataframe
            last_occurrence_f1 = predictions_df.loc[predictions_df['fighter'] == matchup[0]].index[-1]
            last_occurrence_f2 = predictions_df.loc[predictions_df['fighter'] == matchup[1]].index[-1]
        except IndexError:
            # If an IndexError occurs, skip to the next matchup
            continue
        
        # Check if both fighters have more than 2 total fights
        if (
            predictions_df.loc[last_occurrence_f1, 'total_fights'] > 2 and
            predictions_df.loc[last_occurrence_f2, 'total_fights'] > 2
        ):
            # Set rounds to 5 for the first matchup, if it's the final fight or a title fight
            if "Title" in weightclasses[i] or i == 0:
                predictions_df.loc[last_occurrence_f1, 'rounds'] = 5
                predictions_df.loc[last_occurrence_f2, 'rounds'] = 5

            # Set fight numb, weight class, days since last fight and age for both fighters
            predictions_df.loc[last_occurrence_f1, 'fight_num'] = i
            predictions_df.loc[last_occurrence_f2, 'fight_num'] = i
            predictions_df.loc[last_occurrence_f1, 'weight_class'] = weightclasses[i].lower().replace(" ", "_").replace("'", "")
            predictions_df.loc[last_occurrence_f2, 'weight_class'] = weightclasses[i].lower().replace(" ", "_").replace("'", "")
            predictions_df.loc[last_occurrence_f1, 'days_since_last_fight'] = date - predictions_df.loc[last_occurrence_f1, 'date']
            predictions_df.loc[last_occurrence_f2, 'days_since_last_fight'] = date - predictions_df.loc[last_occurrence_f2, 'date']
            dob1 = predictions_df.loc[last_occurrence_f1, 'dob']
            dob2 = predictions_df.loc[last_occurrence_f2, 'dob']
            predictions_df.loc[last_occurrence_f1, 'age'] = (date - dob1).days // 365
            predictions_df.loc[last_occurrence_f2, 'age'] = (date - dob2).days // 365

    # Join the rows in the predictions dataframe and drop excluded columns
    joined_predictions = join_rows(predictions_df, True)
    joined_predictions = joined_predictions.drop(exclude, axis=1)
    # Save the joined predictions dataframe as a CSV file
    joined_predictions.to_csv("data/prediction_data.csv", index=False)


def join_rows(df, predictions=False):
    grouped = df.groupby('fight_num')
    exclude = ['opponent_fight_num', 'opponent_date', 'opponent_result', 'opponent_weight_class', 'opponent_weight_class_code', 
               'opponent_method', 'opponent_time', 'opponent_round_finished', 'opponent_rounds']

    rows = []

    if predictions:
        for _, group in grouped:
            if len(group) >= 2 and group.iloc[0]['fight_num'] < 15:
                row1 = group.iloc[1]
                row2 = group.iloc[0]
                merged_row = pd.concat([row1, row2.rename(lambda x: 'opponent_' + x)])
                merged_row1 = pd.concat([row2, row1.rename(lambda x: 'opponent_' + x)])
                rows.append(merged_row)
                rows.append(merged_row1)
    else:
        for _, group in grouped:
            if len(group) >= 2:
                row1 = group.iloc[0]
                row2 = group.iloc[1]
                merged_row = pd.concat([row1, row2.rename(lambda x: 'opponent_' + x)])
                merged_row1 = pd.concat([row2, row1.rename(lambda x: 'opponent_' + x)])
                rows.append(merged_row)
                rows.append(merged_row1)
    
    new_df = pd.DataFrame(rows).reset_index(drop=True)
    new_df.sort_values(by="fight_num", inplace=True)
    new_df.drop(exclude, axis=1, inplace=True)
    return new_df


def split_dataframe(df):
    details_columns = ['fight_num', 'date', 'fighter', 'result', 'weight_class']
    method_columns = ['method', 'time format', 'time', 'round']
    cols = ['sig_str_landed', 'sig_str_attempts', 'sig_str_received', 'sig_str_avoided', 'sig_str_percent', 'str_landed', 
        'str_attempts', 'str_received', 'str_avoided', 'str_percent', 'td_landed', 'td_attempts', 'td_received', 'td_avoided', 
        'td_percent', 'kd', 'kd_received', 'sub_att', 'rev', 'ctrl', 'sig_reg_mixture', 'sig_reg_percent']
    
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
        
        new_data.append(date_row + fighter_2_row + result_2_row + weightclass_row + method_row + row_2)
        new_data.append(date_row + fighter_1_row + result_1_row + weightclass_row + method_row + row_1)

    new_df = pd.DataFrame(new_data, columns=new_columns)
    return new_df


def reformat_data(df):
    df['method'] = df['method'].replace({
        "TKO - Doctor's Stoppage": "KO/TKO",
        "Decision - Unanimous": "Decision",
        "Decision - Split": "Decision",
        "Decision - Majority": "Decision",
        "Overturned": "Other",
        "DQ": "Other",
        "Could Not Continue": "Other"
    })

    for col in ["sig_str_percent", "td_percent"]:
        df[col] = df[col].str.strip("%").replace("---", "0").fillna(0).apply(lambda x: int(x) / 100)
    
    
    df['date'] = pd.to_datetime(df['date']).fillna(pd.Timedelta(0))
    df['weight_class'] = df['weight_class'].apply(extract_weightclass).str.strip().str.lower().str.replace(" ", "_").replace("'", "")

    # Remove any 'OT' occurrences if present and convert to datetime object  
    df['time'] = pd.to_datetime(df['time'].str.replace('OT', ''), format='%M:%S').apply(convert_to_minutes) # Convert to minutes  
    df['ctrl'] = pd.to_datetime(df['ctrl'], format='%H:%M:%S', errors='coerce').fillna(pd.to_datetime('00:00:00', format='%H:%M:%S')).apply(convert_to_minutes) # Convert to minutes
   
    df = df.rename(columns={'round': 'round_finished'})
    df['rounds'] = df['time format'].apply(lambda x: "1" if x == "No Time Limit" else x).str.extract('^(\d+)', expand=False).astype(int)
    df = df.drop('time format', axis=1)
    return df


def extract_weightclass(expression):
    women_weight_class = re.search(r"\b(?:Women's Strawweight|Women's Flyweight|Women's Bantamweight|Women's Featherweight)\b", expression)
    men_weight_class = re.search(r"\b(?:Flyweight|Bantamweight|Featherweight|Lightweight|Welterweight|Middleweight|Light Heavyweight|Heavyweight|Catch Weight)\b", expression)
    if women_weight_class:
        return women_weight_class.group()
    elif men_weight_class:
        return men_weight_class.group()
    else:
        return "Catch Weight"


def add_new_features(df, step, predcitions=False):
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

    temp_df = pd.DataFrame()
    grouped_fighters = df.groupby('fighter', group_keys=False)
    
    temp_df['fighter'] = df['fighter']
    temp_df['fight_num'] = df['fight_num']
    df['total_fight_time'] = grouped_fighters['time'].cumsum().shift(step)
    temp_df['days_since_last_fight'] = grouped_fighters['date'].apply(lambda x: x - x.shift(step))
    temp_df['total_fights'] = grouped_fighters.cumcount() - step + 1
    
    # Total wins, losses & w/l by decison, ko, sub, other
    for result in ["W", "L"]:
        temp_df["total_" + result.lower()] = grouped_fighters.apply(lambda x: (x['result'] == result).shift(step).cumsum())
        for method in ['Decision','KO/TKO','Other','Submission']:
            temp_df[result.lower() + "_" + method.lower()] = grouped_fighters.apply(lambda x: ((x['result'] == result) & (x['method'] == method)).shift(step).cumsum())

    for col in accuracy:
        # total accuracy score
        temp_df[col + '_accuracy'] = grouped_fighters[col + '_landed'].cumsum().shift(step) / grouped_fighters[col + '_attempts'].cumsum().shift(step)

    for col in differentials:
        differential(df, col)
        
    for col in lags + ['sig_str_landed_differential', 'str_landed_differential']:
        if predcitions:
            for i in range(0, 3):
                lag(df, col, i, predcitions)
        else:
            for i in range(1, 4):
                lag(df, col, i)
        
        calculate_weighted_avg(df, col)
        calculate_avg(df, col)

    for col in totals:
        if col == "round_finished":
            df["foo"] = grouped_fighters[col].cumsum()
            temp_df['total_rounds'] = grouped_fighters['foo'].shift(step)
        else:
            df["foo1"] = grouped_fighters[col].cumsum()
            temp_df["total_" + col] = grouped_fighters['foo1'].shift(step)
            
    df = df.drop(['foo', 'foo1'], axis=1)

    # Iterate over each group
    for _, group in grouped_fighters:
        # Reset streak counters for each group
        current_win_streak = 0
        current_loss_streak = 0
        
        # Iterate over rows in the group
        for index, row in group.iterrows():
            # Update streak columns for the current row
            if step != 0:
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
            if step == 0:
                temp_df.at[index, 'win_streak'] = current_win_streak
                temp_df.at[index, 'loss_streak'] = current_loss_streak
    
    fighters_df = pd.read_csv("data/fighter_stats.csv")

    merged_df = pd.merge(df.copy(), fighters_df, on='fighter')
    merged_df['dob'] = pd.to_datetime(merged_df['dob'])
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['age'] = (merged_df['date'] - merged_df['dob']).dt.days // 365
    # Select columns from df
    merged_df = merged_df[['fight_num', 'fighter', 'height', 'reach', 'stance', 'age', 'dob']]

    # Merge with the original dataframe
    df = df.merge(merged_df, on=['fighter', 'fight_num'])
    df = df.merge(temp_df, on=['fighter', 'fight_num'])
    # Take mean of reach and height to fill na values in the respective weight class
    weight_classes = ['catch_weight', "flyweight", "bantamweight", "featherweight", "lightweight", "welterweight", "middleweight", "light_heavyweight", "heavyweight", "women's_strawweight", "women's_flyweight", "women's_bantamweight", "women's_featherweight"]
    for col in ['reach', 'height']:
        for weight_class in weight_classes:
            sliced_df = df[df['weight_class'] == weight_class]
            df.loc[df['weight_class'] == weight_class, col] = df.loc[df['weight_class'] == weight_class, col].fillna(round(sliced_df[col].mean()))
    # Fill na vals with mean age
    df['age'] = df['age'].fillna(round(df['age'].mean()))
    
    encoder = LabelEncoder()
    df = df[df['result'].isin(['W', 'L'])]
    df.insert(4, 'result_code', encoder.fit_transform(df['result']))
    df['fighter_code'] = encoder.fit_transform(df['fighter'])
    df['weight_class_code'] = encoder.fit_transform(df['weight_class'])
    df['stance_code'] = encoder.fit_transform(df['stance'])

    return df


def calculate_avg(df, column):
    df['avg_' + column + "_per_min"] = df[column] / df['total_fight_time']
    df['avg_' + column + "_per_min"] = df.groupby('fighter')['avg_' + column + "_per_min"].shift(1)


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


def lag(df, column, step, predictions=False):
    if predictions:
        df[column + '_lag' + str(step + 1)] = df.groupby('fighter')[column].shift(step).fillna(0).astype(int)
    else:
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
    K = 32 # Maximum change in rating
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
            row_1['future_elo'] = fighter_ratings.get(group_df.iloc[0]['fighter'], INITIAL_RATING)
            row_1['future_elo_change'] = 0
            row_2['future_elo'] = fighter_ratings.get(group_df.iloc[1]['fighter'], INITIAL_RATING)
            row_2['future_elo_change'] = 0
            
        if not other:
            winner_rating = fighter_ratings.get(winner, INITIAL_RATING)
            loser_rating = fighter_ratings.get(loser, INITIAL_RATING)
            winner_new_rating, loser_new_rating = update_ratings(winner, loser)
            fighter_ratings[winner] = winner_new_rating
            fighter_ratings[loser] = loser_new_rating
            if row_1['fighter'] == winner:
                
                row_1['future_elo'] = winner_new_rating
                row_1['future_elo_change'] = winner_new_rating - winner_rating
                
                row_2['future_elo'] = loser_new_rating
                row_2['future_elo_change'] = loser_new_rating - loser_rating
            elif row_2['fighter'] == winner:
                
                row_1['future_elo'] = loser_new_rating
                row_1['future_elo_change'] = loser_new_rating - loser_rating
                
                row_2['future_elo'] = winner_new_rating
                row_2['future_elo_change'] = winner_new_rating - winner_rating
                
        rows.append(row_1)
        rows.append(row_2)
    
    elo_table = pd.DataFrame(rows)
    group = elo_table.groupby('fighter')
    elo_table['curr_elo'] = group['future_elo'].shift(1)
    elo_table['curr_elo_change'] = group['future_elo_change'].shift(1)

    # Set the first occurrence of each fighter in 'curr_elo' to 1200
    first_occurrences = elo_table.groupby('fighter').head(1).index
    elo_table.loc[first_occurrences, 'curr_elo'] = 1200
    elo_table.loc[first_occurrences, 'curr_elo_change'] = 0
    
    return elo_table


if __name__ == "__main__":
    main()