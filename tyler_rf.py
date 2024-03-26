import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Read the data
day_data = pd.read_csv("day_data.csv")

# Convert holiday column to binary encoding
day_data['holiday'] = day_data['holiday'].apply(lambda x: 0 if pd.isna(x) else 1)

# Convert date column to datetime format
day_data['date'] = pd.to_datetime(day_data['date'])

# Split data into features (X) and target variable (Y)
Y = day_data['arrest']
X = day_data.drop(columns=['arrest', 'date'])

# One-hot encode categorical variables
X = pd.get_dummies(X, prefix='wd')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=24)

# Train Random Forest model
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)

# Evaluate model on testing data
y_pred_rf = random_forest.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Random Forest MSE:", mse_rf)


import shap
# Compute SHAP values
explainer = shap.TreeExplainer(random_forest)
shap_values = explainer.shap_values(X_test)

# Summarize SHAP values
shap.summary_plot(shap_values, X_test, plot_type='bar')
