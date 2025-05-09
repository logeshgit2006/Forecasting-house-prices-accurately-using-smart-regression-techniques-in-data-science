import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ================================================
# 1. Load Data
# ================================================
df = pd.read_excel("dataset (1).xlsx")
df.columns = df.columns.str.strip().str.title()
df.drop_duplicates(inplace=True)
df.dropna(subset=["Bedrooms", "Bathrooms", "Floors", "Garage", "Condition", "Price"], inplace=True)
df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
df.reset_index(drop=True, inplace=True)

# ================================================
# 2. Model Training
# ================================================
features = ["Bedrooms", "Bathrooms", "Floors", "Garage", "Condition", "TotalRooms"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ================================================
# 3. Prediction Function
# ================================================
def predict_price(bedrooms, bathrooms, floors, garage, condition):
    total_rooms = bedrooms + bathrooms
    input_data = pd.DataFrame([{
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Garage": garage,
        "Condition": condition,
        "TotalRooms": total_rooms
    }])
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_data)[0]
    return f"${prediction:,.2f}"

# ================================================
# 4. Gradio Interface
# ================================================
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Floors"),
        gr.Number(label="Garage"),
        gr.Dropdown(["Excellent", "Good", "Fair"], label="Condition")
    ],
    outputs="text",
    title="House Price Predictor",
    description="Enter details to predict the price of a house."
)

iface.launch()
