import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(12, 7)
def plot_predictions(x, y, c="r", label="test"):
    axes[0].scatter(x, y, c=c, s=4, label=label)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))
    
    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias

torch.manual_seed(42)

model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
split = int(0.8*len(X))
x_train, y_train = X[:split], y[:split]
x_test, y_test = X[split:], y[split:]

with torch.inference_mode():
    y_preds = model_0(x_test)

# y_preds = model_0(x_test)

plot_predictions(x_train, y_train, c="b", label="train data")
plot_predictions(x_test, y_test, c='g', label="test data")
plot_predictions(x_test, y_preds, c="r", label="pred1")

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 200
epoch_x = torch.arange(0, epochs)
loss_train = []
loss_preds = []


for epoch in range(epochs):
    model_0.train()

    # print(f"Loss: {loss}")

    optimizer.zero_grad()

    y_pred = model_0(x_train)
    
    # with torch.inference_mode():
    #     y_preds = model_0(x_test)
    #     color = "#%0x%0x%0x" % (epoch*epoch%256, epoch*epoch%256, epoch*epoch%256)
    #     plot_predictions(x_test, y_preds, c=color, label=f"pred-{epoch}")

    loss = loss_fn(y_pred, y_train)
    loss_train.append(loss)
    loss.backward()
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        y_pred = model_0(x_test)
        loss_pred = loss_fn(y_pred, y_test)
        loss_preds.append(loss_pred)

print(model_0.state_dict())

plot_predictions(x_test, y_pred, c="r", label="pred2")

axes[0].legend(prop={"size":14})


axes[1].plot(epoch_x, np.array(torch.tensor(loss_train).numpy()), label="train loss")
axes[1].plot(epoch_x, loss_preds, label="test loss")
axes[1].set_title("training and test loss curves")
axes[1].set_xlabel("epochs")
axes[1].set_ylabel("loss")
axes[1].legend()

plt.show()