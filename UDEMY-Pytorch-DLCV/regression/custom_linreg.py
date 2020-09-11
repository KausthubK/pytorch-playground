import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
DATASET
"""

X = torch.randn(100,1)*10
y = X + 3*torch.randn(100,1)
plt.plot(X.numpy(), y.numpy(), 'o')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


"""
CLASS
"""


class LR(nn.Module):
    # __init__ is the constructor
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred

########
########   SCRIPT
########
torch.manual_seed(1)
model = LR(1,1)
print(model)
[m, b] = model.parameters()
print(m,b)