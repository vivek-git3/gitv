import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Fetch historical stock price data
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Preprocessing
# Use the "Close" price for prediction.
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])  # Ensure that Date is in datetime format

# Features (X) will be the date in ordinal form, and target (y) will be the closing price.
data['Date_ordinal'] = data['Date'].map(lambda x: x.toordinal())  # Convert date to ordinal

X = data[['Date_ordinal']]  # Date (ordinal) as the independent variable
y = data['Close']  # Close price as the dependent variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 7: Visualize the results
plt.figure(figsize=(10, 6))

# Plot the actual stock prices vs the predicted stock prices
plt.plot(data.index, data['Close'], label='Actual Prices', color='blue')

# Map the ordinal values back to datetime for plotting
predicted_dates = pd.to_datetime(X_test['Date_ordinal'], origin='julian', unit='D')

plt.plot(predicted_dates, y_pred, label='Predicted Prices', color='red')

plt.title(f'{ticker} Stock Price Prediction using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
