import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Assuming you have a DataFrame named 'cab_data' with columns 'KM Travelled' and 'Price Charged'
# Make sure 'KM Travelled' is the feature and 'Price Charged' is the target variable
Cab_Data = pd.read_csv('Cab_Data.csv')
# Extracting features and target variable
X = Cab_Data[['KM Travelled']]
y = Cab_Data['Price Charged']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Saving the model using joblib
joblib.dump(model, 'linear_regression_model.pkl')

print("Model trained and saved successfully.")
