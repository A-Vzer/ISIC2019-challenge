from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary

modelPath = 'C:\\Users\Vzer\\PycharmProjects\\ISIC2019\\modelfiles\\efficientnetb3_lr_0.01_bs_16_ep_50_pretr_True_erase.pth'
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=9)
model.load_state_dict(torch.load(modelPath))

x_image = Variable(torch.randn(1, 3, 224, 224)).cuda()
image_modules = list(model.children())[:-1]  # all layer expect last layer
modelA = nn.Sequential(*image_modules).cuda()
out = modelA(x_image)
print(out)