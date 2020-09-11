import torch

x = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
y.backward()
print("Gradient = {}".format(x.grad))
print("Gradient = " + str(x.grad))


x1 = torch.tensor(1.0, requires_grad=True)
z1 = torch.tensor(2.0, requires_grad=True)
y1 = x1**2 + z1**3
y1.backward()
print("X Gradient = {}, Y Gradient = {}".format(x1.grad, z1.grad))
print("X Gradient = " + str(x1.grad) + ", Y Gradient = " + str(z1.grad))
