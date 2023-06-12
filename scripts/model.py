import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import lightgbm as lgb
import joblib
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Read the CSV file
file_path = os.path.join("data", "fights.csv")
training_df = pd.read_csv(file_path)
training_df = training_df.replace([np.inf, -np.inf], 0)

# Define exclusion lists, result columns, model types, and model names
exclude_lists = {
    'sig_str': ['sig_str_landed', 'sig_str_attempts', 'sig_str_received', 'sig_str_avoided', 'sig_str_percent', 'sig_str_landed_differential', 'opponent_sig_str_landed', 
                'opponent_sig_str_attempts', 'opponent_sig_str_received', 'opponent_sig_str_avoided', 'opponent_sig_str_percent', 'opponent_sig_str_landed_differential'
                ],
    'kd': ['kd', 'kd_received', 'opponent_kd', 'opponent_kd_received'],
    'td': ['td_landed', 'td_received', 'td_attempts', 'td_avoided', 'opponent_td', 'opponent_td_received', 'opponent_td_attempts', 'opponent_td_avoided', 
           'td_percent', 'opponent_td_percent'],
    'rev': ['rev', 'opponent_rev',],
    'sub_att': ['sub_att', 'opponent_sub_att',]
}
result_columns = ['fighter_code', 'opponent_fighter_code', 'rounds', 'curr_elo', 'curr_elo_change', 'total_fight_time', 'total_w', 'total_l', 'win_streak', 'loss_streak', 'days_since_last_fight',
    'w_decision', 'w_ko/tko', 'w_other', 'w_submission', 'l_decision', 'l_ko/tko', 'l_other', 'l_submission',
    'opponent_curr_elo', 'opponent_curr_elo_change', 'opponent_total_fight_time', 'opponent_total_w', 'opponent_total_l', 'opponent_win_streak', 
    'opponent_loss_streak', 'opponent_days_since_last_fight',
    'opponent_w_decision', 'opponent_w_ko/tko', 'opponent_w_other', 'opponent_w_submission', 'opponent_l_decision', 'opponent_l_ko/tko', 'opponent_l_other', 'opponent_l_submission',
    'opponent_total_fights', 'opponent_total_rounds', 'total_fights', 'total_rounds', 'weight_class_code']


def main():
    model = train_models()


def train_models():
    model_types = [LinearRegression]
    model_names = ['linearreg']
    train_stats_model(model_types, model_names)
    res_model = train_results_model()
    return res_model


# Function to get training and testing data for a given model
def get_train_test_data(columns, target):
    X = training_df[columns]
    y = training_df[target]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    return X_train, X_test, y_train, y_test


def train_stats_model(model_types, model_names):
    # Loop over the model types and apply the model to each dataset
    for model_type, model_name in zip(model_types, model_names):
        print("-" * 50)
        for feature_type, exclude_list in exclude_lists.items():
            feature_columns = [col for col in training_df.columns if feature_type in col and col not in exclude_list] + result_columns

            if feature_type == "td" or feature_type == "sig_str":
                target_variable = f"{feature_type}_landed"
            else:
                target_variable = f"{feature_type}"
            
            # Get the training and testing data
            X_train, X_test, y_train, y_test = get_train_test_data(feature_columns, target_variable)
            
            # Initialize and fit the model
            model = model_type()
            model.fit(X_train, y_train)
            
            # Make predictions on the testing data
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, a_min=0, a_max=None)
            
            # Create a DataFrame with actual and predicted values
            result_df = pd.DataFrame()
            result_df['actual'] = y_test
            result_df['pred'] = np.round(y_pred)
            
            # Evaluate the performance of the model
            mse = mean_squared_error(y_test, y_pred)
            print(f"{model_name} Mean Squared Error for {feature_type} model:", mse)
            
            # Save the result DataFrame to a CSV file
            file_name = f"{model_name}_{feature_type}_model.csv"
            # joblib.dump(model, f'{model_name}_{feature_type}.pkl')
            # result_df.to_csv(file_name)


def result_data(df, predictions=False):
    if not predictions:
        encoder = LabelEncoder()
        df = df[df['result'].isin(['W', 'L'])]
        res = df.copy()
        res['result_code'] = encoder.fit_transform(df['result'])
    
    features = []
    
    for feature_type, exclude_list in exclude_lists.items():
        x = df[[col for col in df.columns if feature_type in col and col not in exclude_list]]
        features.extend(x)

    X = df[features + result_columns]
    
    if predictions:
        return X
    
    y = res['result_code']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    return X_train, X_test, y_train, y_test


def train_results_model():
    # Instantiate the model
    model_types = [xgb.XGBClassifier, LogisticRegression, lgb.LGBMClassifier, GaussianNB, KNeighborsClassifier, RandomForestClassifier, MLPClassifier, GradientBoostingClassifier, SVC]

    for model_type in model_types:
        model = model_type()
        # X_train, X_test, y_train, y_test = result_data(training_df)
        X_train, X_test, y_train, y_test = result_data(training_df)
        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Concatenate X_test and y_test
        result_df = pd.DataFrame()
        result_df['actual'] = y_test
        result_df['pred'] = y_pred
        
        # Evaluate the performance of the model
        accuracy = accuracy_score(y_test, y_pred)
        model_name = model_type.__name__
        print(f"{model_name} accuracy:", accuracy)
        # result_df.to_csv(f"lgbm_results.csv")
        # joblib.dump(model, 'models/lgbm_results_model.pkl')


if __name__ == "__main__":
    main()