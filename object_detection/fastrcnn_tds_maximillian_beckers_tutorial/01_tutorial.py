
"""
version
"""
import pycocotools
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw
import pandas as pd

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

image = Image.open("/home/kausthubk/working_datasets/raccoon_dataset/images/raccoon-1.jpg")
image.show()

labels = pd.read_csv("/home/kausthubk/working_datasets/raccoon_dataset/data/raccoon_labels.csv")

print(labels.head())


def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin",
                                                      "xmax", "ymax"]].values
    return boxes_array


class RaccoonDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_annot(self.path_to_data_file,self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)

        num_objs = len(box_list)  # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])  # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


dataset = RaccoonDataset(root="/home/kausthubk/working_datasets/raccoon_dataset",
                         data_file="/home/kausthubk/working_datasets/raccoon_dataset/data/raccoon_labels.csv")

datapoint = dataset.__getitem__(0)
print(datapoint)


def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=num_classes)
    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


"""
Dataset Preparation
"""

# use our dataset and defined transformations
dataset = RaccoonDataset(root= "/home/kausthubk/working_datasets/raccoon_dataset",
          data_file= "/home/kausthubk/working_datasets/raccoon_dataset/data/raccoon_labels.csv",
          transforms = get_transform(train=True))

dataset_test = RaccoonDataset(root= "/home/kausthubk/working_datasets/raccoon_dataset",
                              data_file= "/home/kausthubk/working_datasets/raccoon_dataset/data/raccoon_labels.csv",
                              transforms = get_transform(train=False))


# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-40])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-40:])


"""
Dataloaders
"""
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
         dataset_test, batch_size=1, shuffle=False, num_workers=4,
         collate_fn=utils.collate_fn)
print("We have: {} examples, {} are training and {} testing".format(len(indices), len(dataset), len(dataset_test)))


"""
Training the model
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 3
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)

torch.save(model.state_dict(), "/home/kausthubk/out/object_detection/raccoon/model_params.pt")
torch.save(model, "/home/kausthubk/out/object_detection/raccoon/model.pt")


print("\n\nINFERRING FROM MODEL STATE DICT\n\n")
"""
Infer from model_state_dict
"""
loaded_model = get_model(num_classes = 2)
loaded_model.load_state_dict(torch.load( "/home/kausthubk/out/object_detection/raccoon/model_params.pt"))

idx = 0
img, _ = dataset_test[idx]
label_boxes = np.array(dataset_test[idx][1]["boxes"])
loaded_model.eval()
with torch.no_grad():
    prediction = loaded_model([img])
print("\nPrediction: " + str(prediction))
print("\nPrediction Type: " + str(type(prediction)))
image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
draw = ImageDraw.Draw(image)
for elem in range(len(label_boxes)):
    draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]), (label_boxes[elem][2], label_boxes[elem][3])],
                   outline="green", width=3)

for element in range(len(prediction[0]["boxes"])):
    boxes = prediction[0]["boxes"][element].cpu().numpy()
    score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)

    if score > 0.8:
        draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red", width =3)
        draw.text((boxes[0], boxes[1]), text = str(score))

image.show()


"""
Infer from full model
"""
device = torch.device("cuda")
print("\n\nINFERRING FROM FULL MODEL\n\n")
loaded_full_model = torch.load( "/home/kausthubk/out/object_detection/raccoon/model.pt", map_location="cuda:0")
loaded_full_model.to(device)

idx = 0
img, _ = dataset_test[idx]
label_boxes = np.array(dataset_test[idx][1]["boxes"])
loaded_full_model.eval()
with torch.no_grad():
    prediction = loaded_full_model([img.to(device)])
print("\nPrediction: " + str(prediction))
print("\nPrediction Type: " + str(type(prediction)))
image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
draw = ImageDraw.Draw(image)
for elem in range(len(label_boxes)):
    draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]), (label_boxes[elem][2], label_boxes[elem][3])],
                   outline="green", width=3)

for element in range(len(prediction[0]["boxes"])):
    boxes = prediction[0]["boxes"][element].cpu().numpy()
    score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)

    if score > 0.8:
        draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red", width =3)
        draw.text((boxes[0], boxes[1]), text = str(score))

image.show()
