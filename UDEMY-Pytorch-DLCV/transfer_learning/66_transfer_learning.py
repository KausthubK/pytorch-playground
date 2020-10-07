# Dataset: https://github.com/jaddoescad/ants_and_bees

from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
import PIL


'''
DEVICE SELECTION
'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training on device: {}".format(device))


'''
DATA PREPARATION
'''
transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      # ((mean,), (std,)) --> ensuring 1 channel output. for 3 channel use ((m,m,m), (s,s,s)
                                      ])

transform_val = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    # ((mean,), (std,)) --> ensuring 1 channel output. for 3 channel use ((m,m,m), (s,s,s)
                                    ])

training_dataset = datasets.ImageFolder('./ants_and_bees-master/train', transform=transform_train)
validation_dataset = datasets.ImageFolder('./ants_and_bees-master/val', transform=transform_val)

training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False)

print("Num. Training Images: {}".format(len(training_dataset)))
print("Num. Validation Images: {}".format(len(validation_dataset)))

img_0, lab_0 = training_dataset[0]
print(type(img_0))
print(type(lab_0))
print(lab_0)
classes = ('ants', 'bees')

model = models.alexnet(pretrained=True)
print(model)

# freeze feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False

n_inputs = model.classifier[6].in_features
print("Number of input features to last layer = {}".format(n_inputs))
last_layer = nn.Linear(in_features=n_inputs, out_features=len(classes))
model.classifier[6] = last_layer
n_outputs = model.classifier[6].out_features
print("Number of output features to last layer = {}".format(n_outputs))

print(model)
model.to(device=device)

'''
Training
'''

print("____________________")
print("Begin Training on GPU")
print("____________________")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=0.01)  # small LR reduces overstepping/divergence - too small means slow training
epochs = 10  # too few can result in underfitting, too many can result in overfitting

running_loss_history = []
running_corrects_history = []
for e in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    print('')
    for images, labels in tqdm(training_loader, desc='Training Epoch {}'.format(e+1)):
        # print(type(images))
        # print("labels: {}".format(labels))

        inputs = images.to(device)  # set this to device
        labels = labels.to(device)  # set this to device

        outputs = model(inputs)  # infer
        print(labels)
        print(outputs)
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
        epoch_loss = running_loss/len(training_loader.dataset)
        epoch_accuracy = float(running_corrects) / len(training_loader.dataset)
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
