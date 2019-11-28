import torch

x = torch.tensor(2.0,requires_grad = True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
y.backward()
g = x.grad
print(g)


x1 = torch.tensor(1.0, requires_grad=True)
z1 = torch.tensor(2.0, requires_grad=True)
y1 = x1**2 + z1**3
y1.backward()
print(x1.grad, z1.grad)