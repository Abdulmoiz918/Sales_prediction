# Title of Project
# Big Sales Prediction using Random Forest Regressor

# Objective
# Predict the sales revenue of a store based on multiple features such as store size, location, and promotional activity.

# Data Source
# The dataset used is the Big Mart Sales dataset, which contains information about different stores, products, and sales.

# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import Data
# Assuming the dataset is saved as "big_sales_data.csv"
data = pd.read_csv(r'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/B1g%20Sales%20Data.csv')

# Describe Data
print("Data Overview:")
print(data.head())
print("\nData Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], kde=True, color='blue')
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Data Preprocessing
# Handling missing values
data.fillna(data.median(), inplace=True)

# Encoding categorical variables
data = pd.get_dummies(data, drop_first=True)

# Define Target Variable (y) and Feature Variables (X)
X = data.drop(columns=['Sales'])  # Features
y = data['Sales']  # Target variable

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Prediction
sample_data = X_test.iloc[0].values.reshape(1, -1)  # Example: Predicting for the first test sample
predicted_sales = model.predict(sample_data)
print("\nSample Prediction:")
print(f"Predicted Sales for the sample: {predicted_sales[0]}")

# Explanation
# The Random Forest Regressor model was trained using historical sales data.
# It predicts the sales revenue for stores based on features like store size, location, and promotions.
# Evaluation metrics like MSE and R2 score demonstrate the model's performance.
