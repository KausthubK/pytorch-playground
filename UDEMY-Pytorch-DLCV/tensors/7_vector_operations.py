import torch
import matplotlib.pyplot as plt

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([1, 2, 3])

print("Addition: " + str(t1 + t2))

print("Multiplication: " + str(t1*5))

dotprod = torch.dot(t1, t2)
print("Dot Product: " + str(dotprod))

t = torch.linspace(0, 10, 100)
print("Lin space t:" + str(t))
y = torch.exp(t)
x = torch.sin(t)*6000
z = torch.cos(t)*6000

plt.figure()
plt.plot(t.numpy(), y.numpy())
plt.plot(t.numpy(), x.numpy())
plt.plot(t.numpy(), z.numpy())
plt.show()
