import torch
from PIL import Image

## Images
images = ["/home/kausthubk/working_datasets/raccoon_dataset/images/raccoon-1.jpg",
          "/home/kausthubk/working_datasets/raccoon_dataset/images/raccoon-2.jpg",
          "/home/kausthubk/working_datasets/raccoon_dataset/images/raccoon-3.jpg"]
imgs = [Image.open(i) for i in images]
imgs[0].show()
imgs[1].show()
imgs[2].show()

## Single Class
model_1_class = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False, classes=1, force_reload=True)
model_1_class
# results_1 = model_1_class()
# results_1.print()
# results_1.show()


print(dir())
print(dir('model_1_class'))
