#load in initial libraries
from google.colab import files
import pandas as pd

#load in previous data
uploaded = files.upload()
final_previous_data = pd.read_csv("final_previous_data.csv", index_col=0)

uploaded = files.upload()
final_season_data = pd.read_csv("final_season_data.csv", index_col=0)

combined_data = pd.concat([final_previous_data, final_season_data], ignore_index=True)


#forward stats
FWD_stats =['total_points','value', 'goals_scoredALL', 'assistsALL', 'total_pointsALL', 'minutesALL', 'goals_concededALL', 'creativityALL', 'influenceALL', 'threatALL', 'bonusALL', 'bpsALL', 'ict_indexALL', 'red_cardsALL', 'yellow_cardsALL', 'now_costALL', 'GW', 'opponent_strength_overall', 'opponent_strength_attack', 'opponent_strength_defence', 'AV_assists_past5', 'AV_bonus_past5', 'AV_bps_past5', 'AV_creativity_past5', 'AV_goals_conceded_past5', 'AV_goals_scored_past5', 'AV_ict_index_past5', 'AV_influence_past5', 'AV_minutes_past5', 'AV_threat_past5', 'AV_total_points_past5', 'AV_value_past5', 'team_strength_rank', 'opponent_strength_rank']

#midfield stats
MID_stats =['total_points', 'goals_scoredALL', 'assistsALL', 'total_pointsALL', 'minutesALL', 'goals_concededALL', 'creativityALL', 'influenceALL', 'threatALL', 'bonusALL', 'bpsALL', 'ict_indexALL', 'red_cardsALL', 'yellow_cardsALL', 'now_costALL', 'GW', 'opponent_strength_overall', 'opponent_strength_attack', 'opponent_strength_defence', 'AV_assists_past5', 'AV_bonus_past5', 'AV_bps_past5', 'AV_creativity_past5', 'AV_goals_conceded_past5', 'AV_goals_scored_past5', 'AV_ict_index_past5', 'AV_influence_past5', 'AV_minutes_past5', 'AV_threat_past5', 'AV_total_points_past5', 'AV_value_past5', 'team_strength_rank', 'opponent_strength_rank']

#defender stats
DEF_stats =['total_points', 'goals_scoredALL', 'assistsALL', 'total_pointsALL', 'minutesALL', 'goals_concededALL', 'creativityALL', 'influenceALL', 'threatALL', 'bonusALL', 'bpsALL', 'ict_indexALL', 'clean_sheetsALL', 'red_cardsALL', 'yellow_cardsALL', 'now_costALL', 'GW', 'opponent_strength_overall', 'opponent_strength_attack', 'opponent_strength_defence', 'AV_assists_past5', 'AV_bonus_past5', 'AV_bps_past5', 'AV_clean_sheets_past5', 'AV_creativity_past5', 'AV_goals_conceded_past5', 'AV_goals_scored_past5', 'AV_ict_index_past5', 'AV_influence_past5', 'AV_minutes_past5', 'AV_threat_past5', 'AV_total_points_past5', 'AV_value_past5', 'team_strength_rank', 'opponent_strength_rank']

#goalkeeper stats
GK_stats =['total_points', 'total_pointsALL', 'minutesALL', 'goals_concededALL', 'influenceALL', 'bonusALL', 'bpsALL', 'ict_indexALL', 'clean_sheetsALL', 'red_cardsALL', 'yellow_cardsALL', 'now_costALL', 'GW', 'opponent_strength_overall', 'opponent_strength_attack', 'AV_bonus_past5', 'AV_bps_past5', 'AV_clean_sheets_past5', 'AV_goals_conceded_past5', 'AV_ict_index_past5', 'AV_influence_past5', 'AV_minutes_past5', 'AV_penalties_saved_past5', 'AV_saves_past5', 'AV_total_points_past5', 'AV_value_past5', 'team_strength_rank', 'opponent_strength_rank']

combined_data["position"].value_counts()

data_filtered = combined_data[combined_data['AV_minutes_past5'] > 0]

mid_data = data_filtered[data_filtered['position'] == 'MID'][mid_stats]

MID_data.to_csv("MID_DATA.csv")
files.download("MID_DATA.csv")

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from google.colab import files

# Load player data
uploaded = files.upload()

MID_data = pd.read_csv(list(uploaded.keys())[0], index_col=0)

# This handles missing values.
numeric_cols = MID_data.select_dtypes(include=np.number).columns.tolist()
MID_data[numeric_cols] = MID_data[numeric_cols].fillna(MID_data[numeric_cols].median())

# Define training features, x and target y
X = MID_data.drop(['total_points'], axis=1) #drop total_points (for predictions)
y = MID_data['total_points']

# KFold cross-validation, 5 splits for lower rmse
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmse = [] #then store fold

# KFold Cross-Validation loop algorithm
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # XGBoost regressor algorithm, with parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.5,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        n_estimators=1000
    )

    # Model Training.
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )

    # Predict on X validation set
    val_preds = model.predict(X_val)
    # Store the RMSE.
    fold_rmse.append(np.sqrt(mean_squared_error(y_val, val_preds)))

# Print RMSE
print(f"Av KFold RMSE: {np.mean(fold_rmse):.4f}")

# dataset upload
uploaded_test = files.upload()
test_data = pd.read_csv(list(uploaded_test.keys())[0], index_col=0)  # Adjust based on the actual file name
# Similar missing values mitigated
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].median())

feature_columns = X.columns.tolist() #extract features from set


X_test = test_data[feature_columns] # Prepare X test for prediction
# Predict total_points for the testing set uploaded
test_data['predicted_total_points'] = model.predict(X_test)


#final columns to output
final_output = test_data[['id', 'name', 'position', 'predicted_total_points', 'AV_minutes_past5']]

MID_df = final_output

MID_df = MID_df[MID_df['position'] == 'MID']

MID_df = MID_df[MID_df['AV_minutes_past5'] > 10] #get rid of players who have no played an average of over 10 mins over 5 games. 

MID_df = MID_df.sort_values(by='predicted_total_points', ascending=False)

MID_df = MID_df.drop(columns=['AV_minutes_past5'])

# to get the goals predictions, simply change the total_points metric, in each instance, to 'goals_scored'. 