import torch

one_d = torch.arange(0,9)
print(one_d)

two_d = one_d.view(3,3)
print(two_d)

print(two_d.dim())

x = torch.arange(0,18).view(2,3,3)
print(x.shape)
print(x)

print(x[1,1,1])


## matrix multiplicaioni
A = torch.tensor([0,3,5,5,5,2]).view(2,3)
B = torch.tensor([3,4,3,-2,4,-2]).view(3,2)

print(torch.matmul(A,B))

print(A @ B)



