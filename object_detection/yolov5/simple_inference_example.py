import torch
import os

print("LOADING FROM PYTORCH HUB")
# https://pytorch.org/hub/ultralytics_yolov5/

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# print(model)

dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]

# Inference
results = model(imgs)

# Results
results.print()  
results.show()  # or .save()

# Data
print(results.xyxy[0])

os.remove('./yolov5s.pt')
