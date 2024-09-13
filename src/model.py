import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
import requests
from io import BytesIO
from efficientnet_pytorch import EfficientNet

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

class AttentionBlock(nn.Module):
    def __init__(self, in_features, middle_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.middle_features = middle_features
        self.out_features = out_features
        
        self.W1 = nn.Linear(in_features, middle_features)
        self.W2 = nn.Linear(middle_features, out_features)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        attention = self.tanh(self.W1(x))
        attention = self.W2(attention)
        attention_weights = self.softmax(attention)
        weighted_x = x * attention_weights
        return weighted_x

class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnext101_32x8d(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        self.attention = AttentionBlock(2048, 512, 2048)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x.view(x.size(0), x.size(1), -1).permute(0, 2, 1))
        x = x.permute(0, 2, 1).view(x.size(0), 2048, 7, 7)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.dropout(self.fc2(x)))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit

class EfficientNetWithGlobalLocalAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        
        self.global_attention = AttentionBlock(2048, 512, 2048)
        self.local_attention = AttentionBlock(256, 64, 256)
        
        self.conv_local = nn.Conv2d(2048, 256, kernel_size=1)
        
        self.fc1 = nn.Linear(2048 + 256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        features = self.efficientnet.extract_features(x)
        
        global_features = self.global_attention(features.view(features.size(0), features.size(1), -1).permute(0, 2, 1))
        global_features = global_features.permute(0, 2, 1).mean(dim=2)
        
        local_features = self.conv_local(features)
        local_features = self.local_attention(local_features.view(local_features.size(0), local_features.size(1), -1).permute(0, 2, 1))
        local_features = local_features.permute(0, 2, 1).mean(dim=2)
        
        combined_features = torch.cat([global_features, local_features], dim=1)
        
        x = torch.relu(self.dropout(self.fc1(combined_features)))
        x = torch.relu(self.dropout(self.fc2(x)))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit

class DenseNetWithMultiHeadAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.densenet = models.densenet201(pretrained=True)
        self.features = self.densenet.features
        
        self.multi_head_attention = nn.MultiheadAttention(1920, 8)
        
        self.fc1 = nn.Linear(1920, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(1920)
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), features.size(1), -1).permute(2, 0, 1)
        
        attended_features, _ = self.multi_head_attention(features, features, features)
        attended_features = attended_features.permute(1, 2, 0).mean(dim=2)
        
        attended_features = self.layer_norm(attended_features)
        
        x = torch.relu(self.dropout(self.fc1(attended_features)))
        x = torch.relu(self.dropout(self.fc2(x)))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionSEResNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnext = models.resnext101_32x8d(pretrained=True)
        self.features = nn.Sequential(*list(self.resnext.children())[:-2])
        
        self.se_block = SEBlock(2048)
        
        self.inception = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.Conv2d(2048, 256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        ])
        
        self.conv_after_inception = nn.Conv2d(1024, 512, kernel_size=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.features(x)
        x = self.se_block(x)
        
        inception_outputs = []
        for layer in self.inception:
            if isinstance(layer, nn.MaxPool2d):
                inception_outputs.append(nn.Conv2d(2048, 256, kernel_size=1)(layer(x)))
            else:
                inception_outputs.append(layer(x))
        
        x = torch.cat(inception_outputs, dim=1)
        x = self.conv_after_inception(x)
        
        avg_x = self.global_avg_pool(x).view(x.size(0), -1)
        max_x = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_x, max_x], dim=1)
        
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.dropout(self.fc2(x)))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit

class DualPathNetworkWithTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dpn = models.dpn131(pretrained=True)
        
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=2688, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=6)
        
        self.fc1 = nn.Linear(2688, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(2688)

    def forward(self, x):
        features = self.dpn.features(x)
        
        b, c, h, w = features.size()
        features = features.view(b, c, -1).permute(2, 0, 1)
        
        transformed = self.transformer(features)
        
        x = transformed.permute(1, 2, 0).mean(dim=2)
        
        x = self.layer_norm(x)
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.dropout(self.fc2(x)))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit

class HybridConvolutionalTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        
        self.conv_reduce = nn.Conv2d(2560, 512, kernel_size=1)
        
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=6)
        
        self.global_attention = nn.MultiheadAttention(512, 8)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        features = self.efficientnet.extract_features(x)
        features = self.conv_reduce(features)
        
        b, c, h, w = features.size()
        features = features.view(b, c, -1).permute(2, 0, 1)
        
        transformed = self.transformer(features)
        
        attended, _ = self.global_attention(transformed, transformed, transformed)
        
        x = attended.permute(1, 2, 0).mean(dim=2)
        
        x = self.layer_norm(x)
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.dropout(self.fc2(x)))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit

num_classes = 10  
model1 = InceptionSEResNeXt(num_classes)
model2 = DualPathNetworkWithTransformer(num_classes)
model3 = HybridConvolutionalTransformer(num_classes)
model4 = ResNetWithAttention(num_classes)
model5 = EfficientNetWithGlobalLocalAttention(num_classes)
model6 = DenseNetWithMultiHeadAttention(num_classes)