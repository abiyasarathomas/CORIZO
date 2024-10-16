import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
file_path = "Downloads/Minor Project Data set (Stock Price Prediction) (1).csv"  # Adjust path as needed
logger.info("Loading dataset...")
data = pd.read_csv(file_path)

# Preprocess the data
logger.info("Preprocessing data...")

# Check for missing values and handle them (forward fill method used here)
data.fillna(method='ffill', inplace=True)

# Feature Engineering (Assuming your dataset has Date, Open, High, Low, Close, Volume columns)
# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Define features (X) and target (y)
X = data[['Open', 'High', 'Low', 'Volume']]  # Adjust based on your dataset
y = data['Close']  # Target is the closing price

# Split data into training and testing sets
logger.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
logger.info("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
logger.info("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logger.info(f"Model Evaluation:\nMean Squared Error (MSE): {mse}\nR^2 Score: {r2}")

# Plot actual vs predicted prices
logger.info("Plotting Actual vs Predicted stock prices...")
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='orange')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Test Samples')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
