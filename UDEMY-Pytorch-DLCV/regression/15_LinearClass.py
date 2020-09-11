import torch
from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features=1, out_features=1)  # for each input there is one output
print("Model Bias: ")
print(model.bias)
print("Model Weight")
print(model.weight)


x = torch.tensor([[2.0], [3.3]])
print("Model X:")
print(model(x))


y = torch.tensor([[2.0], [3.3]])
print("Model X:")
print(model(x))
