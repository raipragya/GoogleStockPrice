Stock Price Prediction using Recurrent Neural Network (RNN)

Introduction

This project demonstrates the use of a Recurrent Neural Network (RNN) to predict the stock prices of a company. The RNN is implemented using LSTM layers for long-term temporal dependencies and is trained on Google's stock price dataset.
Project Overview

The main objective of this project is to:

    Process stock price data.
    Create a predictive model using an RNN.
    Visualize the performance of the trained model in forecasting future stock prices.

Requirements

To run this project, you need the following dependencies:

    Python 3.x
    Numpy
    Pandas
    Matplotlib
    Scikit-learn
    Keras

Installation

    Clone the repository using git clone


Install dependencies:

    pip install -r requirements.txt

    Place the dataset file (Google_Stock_Price_Train.csv) in the root directory of the project.

Dataset

The project uses Google's stock price data for training. Ensure the dataset has at least:

    Date column
    Open price

The training data includes only the 'Open' prices for simplicity.
Model Architecture

The RNN is built with the following layers:

    Input Layer: Accepts a 3D tensor (batch, timesteps, feature).
    Four LSTM Layers: Each with 50 memory units and a 20% Dropout for regularization.
    Dense Output Layer: Outputs the predicted stock price.

The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function.
Training Process

    Preprocessing:
        Feature scaling using MinMaxScaler.
        Creating data structures with 60 timesteps and 1 output.

    Training:
        Train the RNN model with 100 epochs and a batch size of 32.

Usage

Run the project with the following command:

python GoogleStockPricePredictor.py

This script preprocesses the data, trains the RNN, and displays the training loss at each epoch.
Results

The RNN model successfully predicts stock price trends with low error (as shown by the training loss). Use visualization tools like Matplotlib to plot actual vs. predicted prices for better insights.

