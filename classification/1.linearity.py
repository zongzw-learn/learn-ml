from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from __init__ import *
from models import CircleModuleV0

# 1. Make classification data and get it ready
n_samples = 1000

plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu, linewidths=1)
# plt.show()

print(X.shape, y.shape)

# 1.2 Turn data into tensors and create train and test splits
import torch

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X.shape, y.shape)
print(X[:5], y[:5])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# 2. Build a model
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_0 = CircleModuleV0().to(device=device)
print(model_0)

# https://playground.tensorflow.org/

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

print(model_0)

untrained_preds = model_0(X_test.to(device))
print(f"x pred 10: {untrained_preds[:10]}")

# 2.1 Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# 3. Train model
y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
print(y_preds.squeeze())

# 3.2 Building a training and testing loop
torch.manual_seed(42)

epochs = 500

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | "
              f"Loss: {loss: .5f}, Accuracy: {acc: .5f}% | "
              f"Test loss: {test_loss: .5f}, Test acc: {test_acc: .2f}%")

# import requests
# from pathlib import Path 

# # Download helper functions from Learn PyTorch repo (if not already downloaded)
# if Path("helper_functions.py").is_file():
#   print("helper_functions.py already exists, skipping download")
# else:
#   print("Downloading helper_functions.py")
#   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#   with open("helper_functions.py", "wb") as f:
#     f.write(request.content)


from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()