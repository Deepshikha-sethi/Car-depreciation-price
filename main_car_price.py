import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Car Depreciated Selling Price Predictor")
st.write("Enter the age of the car to predict its selling price.")

# Load dataset
@st.cache_data  # caching to speed up loading
def load_data():
    return pd.read_csv("cardata2.csv")

data = load_data()

# Show data preview
if st.checkbox("Show raw data"):
    st.write(data)

# Extract features and target variable
X = data[['age','Fuel type']]
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Input
age_of_car = st.number_input("Enter age of car", min_value=0.0, step=0.1)

# Predict Button
if st.button("Predict Price"):
    pred_price = model.predict([[age_of_car]])
    st.success(f"Predicted price: ₹{pred_price[0]:,.2f}")

# Evaluate the model
if st.button("Evaluate the Model"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R-squared Score:** {r2:.2f}")
