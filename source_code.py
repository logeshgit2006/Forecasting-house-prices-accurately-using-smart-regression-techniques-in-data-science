pip install pandas numpy scikit-learn xgboost gradio matplotlib seaborn

import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('train.csv')

# Handle missing values
data.fillna(data.median(numeric_only=True), inplace=True)

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data)

# Separate features and target variable
X = data.drop('SalePrice', axis=1)
y = np.log1p(data['SalePrice'])  # Log-transform the target for better performance

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost Model MSE: {mse}")
print(f"XGBoost Model R-squared: {r2}")

import gradio as gr

# Define the prediction function
def predict_house_price(OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]],
                              columns=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'])
    
    # Align the input data with the training data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Predict and return the house price
    prediction = xgb_model.predict(input_data)
    return f"${np.expm1(prediction[0]):,.2f}"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_house_price,
    inputs=[
        gr.Slider(1, 10, step=1, label="Overall Quality"),
        gr.Slider(500, 5000, step=50, label="Above Grade Living Area (sq ft)"),
        gr.Slider(0, 4, step=1, label="Garage Cars"),
        gr.Slider(0, 3000, step=50, label="Total Basement Area (sq ft)"),
        gr.Slider(0, 4, step=1, label="Full Bathrooms"),
        gr.Slider(1900, 2025, step=1, label="Year Built")
    ],
    outputs="text",
    title="House Price Predictor",
    description="Enter the details of the house to predict its price."
)

# Launch the interface
interface.launch()
