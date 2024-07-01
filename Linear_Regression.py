import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import ML_OEL

# Load the dataset
df = pd.read_csv('smartphonesDataset.csv')

# Data Preprocessing
df.fillna({'RAM': df['RAM'].mode()[0], 'Storage': df['Storage'].mode()[0]}, inplace=True)

# Feature Engineering: Creating Price_per_GB
if 'Price' in df.columns:
    df['Price_per_GB'] = df['Price'] / df['Storage']

# Feature Scaling
scaler = StandardScaler()
df[['RAM', 'Storage']] = scaler.fit_transform(df[['RAM', 'Storage']])

# Save the modified dataset
df.to_csv('smartphonesDataset_modified.csv', index=False)

# Select features and target variable
features = ['RAM', 'Storage', 'Price_per_GB']
target = 'Price'

X = df[features]
y = df[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
