import torch
import matplotlib.pyplot as plt

t1 = torch.tensor([1,2,3])
t2 = torch.tensor([1,2,3])

print(t1 + t2)

print(t1*5)

dotprod = torch.dot(t1,t2)
print(dotprod)

t = torch.linspace(0,10,20)
y = torch.exp(t)
x = torch.sin(t)*6000

plt.figure()
plt.plot(t.numpy(),y.numpy())
plt.plot(t.numpy(),x.numpy())
plt.show()

