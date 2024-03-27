import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import shap

day_data = pd.read_csv('day_data.csv')
day_data['date'] = pd.to_datetime(day_data['date'], format="%Y-%m-%d")

train_data = day_data[day_data['date'] < '2024-01-01']
test_data = day_data[(day_data['date'] >= '2024-01-01') & (day_data['date'] <= '2024-01-31')]

features = ['moon', 'arrest', 'wPC1', 'wPC2', 'wPC3']
target = 'num_crimes'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# RMSE
rmse = sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error (RMSE) for Random Forest:", rmse)

# SHAP 
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')
