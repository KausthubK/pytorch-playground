from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))  # ((mean,), (std,)) --> ensuring 1 channel output. for 3 channel use ((m,m,m), (s,s,s)
                                ])

training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
print(training_dataset)  # prints dataset information

training_loader = DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)


def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)  # 1 = colour, 2 = width, 0 = height (1, 28, 28)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image


dataiter = iter(training_loader)
images, labels = dataiter.next()

fig = plt.figure(figsize=(25, 6))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title(labels[idx].item())
plt.show()


class MyClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_1, hidden_layer_2, n_classes):
        super().__init__()
        output_size = n_classes
        self.linear_1 = nn.Linear(input_size, hidden_layer_1)
        self.linear_2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.linear_3 = nn.Linear(hidden_layer_2, output_size)

    def forward(self, x):
        x = relu(self.linear_1(x))
        x = relu(self.linear_2(x))
        x = self.linear_3(x)
        return x


model = MyClassifier(input_size=(1*28*28), hidden_layer_1=125, hidden_layer_2=65, n_classes=10)
print(model)


# Training
print("____________________")
print("Begin Training on CPU")
print("____________________")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=0.01)  # small LR reduces overstepping/divergence - too small means slow training
epochs = 6  # too few can result in underfitting, too many can result in overfitting

running_loss_history = []
running_corrects_history = []
for e in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    print('')
    for images, labels in tqdm(training_loader, desc='Training Epoch {}'.format(e+1)):
        inputs = images.view(images.shape[0], -1)  # flattens colour channel in index 0 is left alone

        outputs = model(inputs)  # infer
        loss = criterion(outputs, labels)  # calculate loss

        # model update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)  # prediction made by the model based on the highest likelihood
        running_corrects += torch.sum(preds == labels.data)
    else:
        # this happens after the for loop finishes
        epoch_loss = running_loss/len(training_loader)
        epoch_accuracy = float(running_corrects) / len(training_loader)
        running_loss_history.append(epoch_loss)
        print('epoch: {}/{} | training loss: {:4f} | accuracy: {:4f}'.format(e+1, epochs, epoch_loss, epoch_accuracy))

print("Training Complete... saving model...", end='')
torch.save(model, './model_{}.pth'.format(int(dt.timestamp(dt.now()) * 1000)))
print('... saved!')

plt.figure()
plt.plot(running_loss_history)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.show()
