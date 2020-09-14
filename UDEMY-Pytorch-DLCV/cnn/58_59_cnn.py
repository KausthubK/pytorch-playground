from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training on device: ")
print(device)

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


class MyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4*4*50, out_features=500)
        self.dropout_1 = nn.Dropout(0.5)  # adding dropout layers can be one strategy of overcoming overfitting
        self.fc_2 = nn.Linear(in_features=500, out_features=n_classes)

    def forward(self, x):
        out_conv1 = relu(self.conv_1(x))
        out_conv1 = max_pool2d(out_conv1, 2, 2)

        out_conv2 = relu(self.conv_2(out_conv1))
        out_conv2 = max_pool2d(out_conv2, 2, 2)

        flatten = out_conv2.view(-1, 4*4*50)

        out_fc1 = relu(self.fc_1(flatten))
        out_dropout1 = self.dropout_1(out_fc1)

        out_fc2 = self.fc_2(out_dropout1)  # no activation function since we're using cross-entropy loss
        return out_fc2


model = MyCNN(n_classes=10).to(device=device)
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
        inputs = images.to(device)  # set this to device
        labels = labels.to(device)  # set this to device

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
