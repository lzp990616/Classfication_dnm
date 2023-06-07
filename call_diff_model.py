import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models

# model = models.densenet121()
# model = models.resnet50()
# model = models.xxx

# such as:
# esnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet = models.mobilenet_v2()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()

model = models.inception_v3()

# vgg16 last layer:
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# then we can change the Linear as the follow:
print(model)  # original network

num_class = 2
# Names may vary from network to network, so it is best to print out the names of the layers.

num_labels = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_labels, num_class)

print(model)
# dnm
model.classifier[6] = nn.Sequential(
    nn.Linear(num_labels, 128),  # hidden layer
    DNM_Linear(input_size=128, out_size=2, M=10, activation=None)
)

# or
model.classifier[6] = DNM_Linear(input_size=num_labels, out_size=2, M=10, activation=None)
# model.fc = xxx  # Inception3 's last layer
