import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import lightgbm as lgb

# Read the CSV file
training_df = pd.read_csv("data/fights.csv")

# Define exclusion lists, result columns, model types, and model names
exclude_lists = {
     'sig_str': ['sig_str_landed', 'sig_str_attempts', 'sig_str_received', 'sig_str_avoided', 'sig_str_percent', 'sig_str_landed_differential', 'opponent_sig_str_landed', 
                    'opponent_sig_str_attempts', 'opponent_sig_str_received', 'opponent_sig_str_avoided', 'opponent_sig_str_percent', 'opponent_sig_str_landed_differential'],
     'kd': ['kd', 'kd_received', 'opponent_kd', 'opponent_kd_received'],
     'td': ['td_landed', 'td_received', 'td_attempts', 'td_avoided', 'opponent_td', 'opponent_td_received', 'opponent_td_attempts', 'opponent_td_avoided', 'td_percent', 'opponent_td_percent'],
     'rev': ['rev', 'opponent_rev',],
     'sub_att': ['sub_att', 'opponent_sub_att',]
}
result_columns = ['rounds', 'total_fight_time', 'win_streak', 'loss_streak', 'days_since_last_fight', 'total_w', 'total_l', 'opponent_total_w', 'opponent_total_l',
     'opponent_total_fight_time', 'opponent_win_streak', 'opponent_loss_streak', 'opponent_days_since_last_fight', 'reach', 'age', 'opponent_reach', 'opponent_age']


def main():
     # Read prediction data and existing predictions
     pred_df = pd.read_csv("data/prediction_data.csv")
     existing_predictions = pd.read_csv('data/predictions.csv')

     # Convert string columns to numeric values
     pred_df['days_since_last_fight'] = pd.to_numeric(pred_df['days_since_last_fight'].str.replace(' days', ''))
     pred_df['opponent_days_since_last_fight'] = pd.to_numeric(pred_df['opponent_days_since_last_fight'].str.replace(' days', ''))

     # Train models and obtain predictions
     res_model, models = train_models()
     X = result_data(pred_df, predictions=True)
     pred = res_model.predict(X)
     proba = res_model.predict_proba(X)

     # Transform prediction values and calculate confidence
     transformed_array = np.where(pred == 0, 'L', 'W')
     confidence = []
     for i, pre in enumerate(pred):
          confidence.append(round(proba[i][pre], 2))

     # Create dictionary to store results
     results = {
     'fight_num': pred_df['fight_num'],
     'fighter': pred_df['fighter'],
     'prediction': transformed_array,
     'prediction_code': pred
     }

     # Obtain additional statistics predictions from models
     for i, (cols, model) in enumerate(models):
          stat_X = get_train_test_data(cols, None, True)
          stat_pred = model.predict(pred_df[stat_X.columns.to_list()])
          results[list(exclude_lists.keys())[i]] = stat_pred

     # Create DataFrame from results and add confidence column
     predictions = np.round(pd.DataFrame(results))
     predictions.insert(3, 'confidence', confidence)

     # Concatenate existing predictions if available
     if existing_predictions is not None:
          predictions = pd.concat([existing_predictions, predictions])

     # Save predictions to CSV file
     predictions.to_csv("data/predictions.csv", index=False)


def train_models():
     model_types = [LinearRegression]
     model_names = ['linearreg']
     models = []
     for feature_columns, model in train_stats_model(model_types, model_names):
          models.append((feature_columns, model))
          
     res_model = train_results_model()
     return res_model, models


# Function to get training and testing data for a given model
def get_train_test_data(columns, target=None, prediction=False):
     X = training_df[columns]
     if prediction:
          return X
     y = training_df[target]
          
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

     return X_train, X_test, y_train, y_test


def train_stats_model(model_types, model_names):
     # Loop over the model types and apply the model to each dataset
     for model_type, model_name in zip(model_types, model_names):
          for feature_type, exclude_list in exclude_lists.items():
               feature_columns = [col for col in training_df.columns if feature_type in col and col not in exclude_list and 'lag' not in col and 'avoided' not in col] + result_columns
          
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
               yield feature_columns, model


def result_data(df, predictions=False):
     if not predictions:
          encoder = LabelEncoder()
          df = df[df['result'].isin(['W', 'L'])]
          res = df.copy()
          res['result_code'] = encoder.fit_transform(df['result'])

     features = []

     for feature_type, exclude_list in exclude_lists.items():
          x = df[[col for col in df.columns if feature_type in col and col not in exclude_list and 'lag' not in col and 'attempts' not in col and 'avoided' not in col and 'received' not in col]]
          features.extend(x)

     if predictions:
          return df[features + result_columns]

     X = df[features + result_columns]
     y = res['result_code']

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

     return X_train, X_test, y_train, y_test


def train_results_model():
     # Instantiate the model
     model_types = [lgb.LGBMClassifier]

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
     return model


if __name__ == "__main__":
     main()