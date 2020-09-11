import torch

# y = mx + b # predicting

m = torch.tensor(3.0, requires_grad=True)  # random init m = 3
b = torch.tensor(1.0, requires_grad=True)  # random init b = 1


def forward(x):
    y = m*x + b
    return y


x1 = torch.tensor(2)
print(forward(x1))

x2 = torch.tensor([[4], [7]])
print(forward(x2))