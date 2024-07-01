import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time

from torch.utils.data import DataLoader, Dataset


def add_gaussian_noise(data, mean=0, std=1):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise



def generate_multidimensional_gaussian_noise(shape, mean=1500, std=1.0):
    """
    生成具有指定形状、均值和标准差的多维高斯噪声。
    :param shape: 数组形状
    :param mean: 高斯分布的均值
    :param std: 高斯分布的标准差
    :return: 高斯噪声数组
    """
    gaussian_noise = np.random.normal(mean, std, size=shape)
    return gaussian_noise


# Read data
data = fetch_california_housing()
X, y = data.data, data.target

x_path = sys.argv[1]
y_path = sys.argv[2]

X = pd.read_csv(x_path)
y = pd.read_csv(y_path)
x_noise = generate_multidimensional_gaussian_noise(X.shape)[:32]
y_noise = generate_multidimensional_gaussian_noise(y.shape)[:32]

Y = np.concatenate((y_noise, y), axis=0)
X = np.concatenate((x_noise, X), axis=0)

print("X: ", X)
print("Y: ", y)
Y = Y
X = torch.tensor(X)
y = torch.tensor(Y)

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(2, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100  # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    # print(X_test,y_pred)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)

time_start = time.time()
x_test = [
    # [0, 4, 4],
    # [1, 285, 16],
    # [1, 188, 18],
    # [1, 10, 8],
    # [1, 18, 10],
    # [1, 17, 10]
    [1, 24],
    [0, 4],
    [1, 15],
    [1, 17],
    [1, 5],
    [1, 404],
]
x_test = torch.tensor(x_test, dtype=torch.float32)
y_pred = model(x_test)

print("PRED: ", x_test)
print("PRED: ", y_pred)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
time_end = time.time()

print("time: ", time_end - time_start, "s")
# plt.plot(history)
# plt.show()
