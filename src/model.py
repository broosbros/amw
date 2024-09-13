import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
import requests
from io import BytesIO

class EntityExtractionModel(nn.Module):
    def __init__(self, num_classes):
        super(EntityExtractionModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_classes) 
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit