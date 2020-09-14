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


def plot_fit(title):
    plt.title(title)
    m1, b1 = model.get_params()
    x1 = np.array([-30, 30])
    y1 = m1*x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X, y)
    plt.show()

plot_fit(title='Tutorial 16')
# fit is pretty bad with this - need gradient descent on a loss function to fit properly


print("____________________")
print("Begin Training")
print("____________________")
# Training

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)  # small LR reduces overstepping/divergence - too small means slow training
epochs = 100  # too few can result in underfitting, too many can result in overfitting

losses = []
for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    print("epoch: {}/{} | loss: {}".format(i+1, epochs, loss.item()))

    losses.append(loss)
    optimizer.zero_grad()  # set gradients to zero before optimization - since gradients accumulate
    loss.backward()  # backpropagate loss
    optimizer.step()  # update model parameters

plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training Progress")
plt.show()

plot_fit(title="Trained Model")
