import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# DATASET
X = torch.randn(100, 1)*10  # random dataset
y = X + 3*torch.randn(100, 1)  # normal distribution of noise
plt.plot(X.numpy(), y.numpy(), 'o')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Dataset')
plt.show()


# custom helper classes
class CustomLinearReg(nn.Module):
    # __init__ is the constructor
    def __init__(self, input_size, output_size):
        print("Initializing LR Class instance")
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, a):
        pred = self.linear(a)
        return pred

    def get_params(self):
        [m_mod, b_mod] = self.parameters()
        return m_mod[0][0].item(), b_mod[0].item()


# SCRIPT
print("____________________")
torch.manual_seed(1)
model = CustomLinearReg(1, 1)

print("____________________")
print("Model:")
print(model)
[m, b] = model.parameters()
print(m)
print(b)


m, b = model.get_params()
print("Model m: {} | b: {}".format(m, b))


# INFERENCE
print("____________________")
print("Test Inference")
print("____________________")
x = torch.tensor([[1.0], [2.0]])
print(model.forward(x))


def plot_fit(title, input_x1):
    plt.title(title)
    m1, b1 = model.get_params()
    y1 = m1*input_x1 + b1
    plt.plot(input_x1, y1, 'r')
    plt.scatter(X, y)
    plt.show()


x1 = np.array([-30, 30])
plot_fit(title='Tutorial 16', input_x1=x1)
# fit is pretty bad with this - need gradient descent on a loss function to fit properly
