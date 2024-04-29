import torch
import torch.nn as nn

x = torch.ones(3, requires_grad=True)

lossfunc = nn.L1Loss()

optifunc = torch.optim.SGD([x], lr=0.5)

torch.manual_seed(42)

y = 0.5+torch.rand(3)

print(f"x: {x}, y: {y}")

loss = lossfunc(x, y)

print(f"before loss.backward, x.grad: {x.grad}")
loss.backward()
print(f"after  loss.backward, x.grad: {x.grad}")

optifunc.step()
print(f"after optimizer.step: x is {x}")