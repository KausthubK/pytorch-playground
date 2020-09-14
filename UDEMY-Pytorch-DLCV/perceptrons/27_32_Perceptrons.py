import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

n_pts = 100
centres = [[-0.5, 0.5], [0.5, -0.5]]  #, [0.95, 0.95]]
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centres, cluster_std=0.45)  # random state is a random seed for reproducibility
x_data = torch.Tensor(X)
y_data = torch.Tensor(y)

print("____________\nDatapoints:\n")
print(X)

print("____________\nLabels:\n")
print(y)


def scatter_plot():
    plt.scatter(X[y == 0, 0], X[y == 0, 1])  # plot class 0
    plt.scatter(X[y == 1, 0], X[y == 1, 1])  # plot class 1
    # plt.scatter(X[y == 2, 0], X[y == 2, 1])  # plot class 2
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


plt.figure()
scatter_plot()


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        print("Initializing Module Class instance")
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, a):
        pred = torch.sigmoid(self.linear(a))
        return pred

    def predict(self, a):
        pred = self.forward(a)
        if pred >= 0.5:
            return 1
        else:
            return 0

    def get_params(self):
        w, b = self.parameters()
        w1, w2 = w.view(2)
        return w1.item(), w2.item(), b[0].item()


torch.manual_seed(1)
model = MyModel(2, 1)
# print(list((model.parameters())))
print(model.get_params())


def plot_fit(title):
    plt.title(title)
    w1, w2, b1 = model.get_params()
    x1 = np.array([-2.0, 2.0])
    x2 = (w1*x1 + b1)/-w2
    plt.plot(x1, x2, 'r')
    scatter_plot()


plt.figure()
plot_fit(title="Tutorial 27 - Untrained Model")


# Training
print("____________________")
print("Begin Training")
print("____________________")

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)  # small LR reduces overstepping/divergence - too small means slow training
epochs = 10000  # too few can result in underfitting, too many can result in overfitting

losses = []
for i in range(epochs):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
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

plot_fit(title="Tutorial 27 - Trained Model")

0
# Testing
print("____________________")
print("Begin Testing")
print("____________________")

plt.figure()
pt1 = torch.Tensor([1.0, -1.0])
pt2 = torch.Tensor([-1.0, 1.0])
plt.plot(pt1.numpy()[0], pt1.numpy()[1], 'rx')
plt.plot(pt2.numpy()[0], pt2.numpy()[1], 'kx')
print("Red cross positive probability = {}".format(model.forward(pt1).item()))
print("Black cross positive probability = {}".format(model.forward(pt2).item()))
print("Red cross class prediction = {}".format(model.predict(pt1)))
print("Black cross class prediction = {}".format(model.predict(pt2)))
plot_fit("Inference on unseen data with trained Model")
