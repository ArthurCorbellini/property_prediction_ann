
import torch.optim as optim
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from neural_network_model import NeuralNetworkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(torch.cuda.get_device_name(device=device))

url = 'https://raw.githubusercontent.com/LeakyAI/FirstNeuralNet/main/lemons.csv'
df = pd.read_csv(url)

# Calculate the mean and standard deviation of the price column, then standardize the price column
priceMean = df['Price'].mean()
priceStd = df['Price'].std()
df['Price'] = (df['Price']-priceMean)/priceStd

# Calculate the mean and standard deviation of the numSold column, then standardize numSold
numSoldMean = df['NumberSold'].mean()
numSoldStd = df['NumberSold'].std()
df['NumberSold'] = (df['NumberSold']-numSoldMean)/numSoldStd

# Create our PyTorch tensors and move to CPU or GPU if available
# Extract the inputs and create a PyTorch tensor x (inputs)
inputs = ['Weekend', 'Sunny', 'Warm', 'BigSign', 'Price']
outputs = ['NumberSold']


# print(df.head(10))
print(df[inputs].head(10))
print(df[outputs].head(10))

nnm = NeuralNetworkModel(df[inputs], df[inputs], df[outputs], df[outputs])
