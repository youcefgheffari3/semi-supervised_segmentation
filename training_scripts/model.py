import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegNet(nn.Module):
    def __init__(self, num_classes=21):  # 21 classes like VOC
        super(SimpleSegNet, self).__init__()
        # Encoder (downsample)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (upsample)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        # Decode
        x = F.relu(self.upconv1(x))
        x = self.conv3(x)  # logits per class
        return x
