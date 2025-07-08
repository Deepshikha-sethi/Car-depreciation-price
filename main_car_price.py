import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create the dataset
data=pd.read_csv("cardata2.csv")
print(data)

# Extract features and target variable
X = data[['age']].values
y = data['Price'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict on the training data
y_pred = model.predict(X)
st.title("Car depreciated selling price Predictor")
st.write("Enter age of car")
age_of_car=st_number_input("enter age of car",min_value=0.0, step=0.1)

#Predict Button
if st.button("Predict Price"):
    pred_price=model.predict([[age_of_car]])
    st.success(f"Predicted price: {pred_price:.2f}")
    
# Evaluate the model
if st.button("Evaluate the model"):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.success(f"Mean Squared Error:",mse)
    st.success("R-squared:", r2)

"""
# Plotting the data and the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Car age')
plt.ylabel('Price')
plt.title('Car age vs Price')
plt.show()"""


