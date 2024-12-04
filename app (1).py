import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset and preprocess
df = pd.read_csv('GOOG.csv')
df = df.drop(columns=[
    'symbol', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'
], axis=1)

# Split the data
X = df[['open', 'high', 'low', 'volume']].values
y = df['close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Define the prediction function
def predict_stock_price(open_price, high_price, low_price, volume):
    input_data = np.array([[open_price, high_price, low_price, volume]])
    predicted_price = regressor.predict(input_data)[0]
    return round(predicted_price, 2)

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_stock_price,
    inputs=[
        gr.Number(label="Opening Price"),
        gr.Number(label="High Price"),
        gr.Number(label="Low Price"),
        gr.Number(label="Volume")
    ],
    outputs=gr.Text(label="Predicted Closing Price"),
    title="Stock Price Prediction App",
    description="Enter stock data to predict the closing price using a Linear Regression model."
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()
