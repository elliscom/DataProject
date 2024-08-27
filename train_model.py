import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np 

DATASET_PATH = 'C:/Users/ellis/OneDrive/Desktop/fpl_app/static/modeldata/largefpldata.csv'

df = pd.read_csv(DATASET_PATH)


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(['overall_rank', 'fpl_id', 'event'], axis=1)
y = df['overall_rank']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train_scaled, y_train)


MODEL_PATH = 'C:/Users/ellis/OneDrive/Desktop/fpl_app/static/modeldata/model.joblib'
SCALER_PATH = 'C:/Users/ellis/OneDrive/Desktop/fpl_app/static/modeldata/scaler.joblib'
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Been saved")
