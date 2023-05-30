import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

# Read the CSV file
file_path = os.path.join("data", "fights.csv")
df = pd.read_csv(file_path)

df = df.replace([np.inf, -np.inf], 0)
# Define exclusion lists, result columns, model types, and model names
exclude_lists = {
    'sig_str': ['sig_str_landed', 'sig_str_attempts', 'sig_str_received', 'sig_str_avoided', 'sig_str_percent', 'sig_str_landed_differential', 'sig_str_accuracy', 'opponent_sig_str_landed', 'opponent_sig_str_attempts', 'opponent_sig_str_received', 'opponent_sig_str_avoided', 'opponent_sig_str_percent', 'opponent_sig_str_landed_differential', 'opponent_sig_str_accuracy'],
    'kd': ['kd', 'kd_received', 'opponent_kd', 'opponent_kd_received'],
    'td': ['td_landed', 'td_received', 'td_attempts', 'td_avoided', 'td_accuracy', 'opponent_td_accuracy', 'opponent_td', 'opponent_td_received', 'opponent_td_attempts', 'opponent_td_avoided', 'td_percent', 'opponent_td_percent'],
    'rev': ['rev', 'opponent_rev'],
    'sub_att': ['sub_att', 'opponent_sub_att']
}
result_columns = ['rounds', 'elo', 'elo_change', 'total_fight_time', 'num_fights', 'total_w', 'total_l', 'total_rounds', 'win_streak', 'loss_streak', 'days_since_last_fight',
     'opponent_elo', 'opponent_elo_change', 'opponent_total_fight_time', 'opponent_num_fights', 'opponent_total_w', 'opponent_total_l', 'opponent_total_rounds', 'opponent_win_streak', 
     'opponent_loss_streak', 'opponent_days_since_last_fight']

model_types = [LinearRegression]
model_names = ['linearreg']

# Function to get training and testing data for a given model
def get_train_test_data(columns, target):
    X = df[columns]
    y = df[target]
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
     
    return X_train, X_test, y_train, y_test

# Loop over the model types and apply the model to each dataset
for model_type, model_name in zip(model_types, model_names):
     print("-" * 50)
     for feature_type, exclude_list in exclude_lists.items():
          feature_columns = [col for col in df.columns if feature_type in col and col not in exclude_list] + result_columns
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
          result_df.to_csv(file_name)

def result_model(df):
     
     features = []
     for feature_type, exclude_list in exclude_lists.items():
          x = df[[col for col in df.columns if feature_type in col and col not in exclude_list]]
          features.extend(x)
     X = df[features + result_columns]
     y = df['result_code']
     
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
     
     return X_train, X_test, y_train, y_test

# model_funcs = [result_model]
# model_types = [lgb.LGBMClassifier]
# model_names = ['lgbm']

# for model_func in model_funcs:
#      for i in range(len(model_types)):
#      # Instantiate the model
#           model = model_types[i]()

#           X_train, X_test, y_train, y_test = model_func(df)
#           # Fit the model on the training data
#           model.fit(X_train, y_train)

#           # Make predictions on the testing data
#           y_pred = model.predict(X_test)

#           # Concatenate X_test and y_test
#           result_df = pd.DataFrame()
#           result_df['actual'] = y_test
#           result_df['pred'] = y_pred
          
#           model_stat = model_func.__name__
#           # Evaluate the performance of the model on the testing data
#           accuracy = accuracy_score(y_test, y_pred)
#           print(f"{model_names[i]} Accuracy {model_stat}:", accuracy)

#           # Save the result DataFrame to a CSV file
#           result_df.to_csv(f"{model_names[i]}_{model_stat}.csv")
#           joblib.dump(model, f'{model_names[i]}_{model_stat}.pkl')