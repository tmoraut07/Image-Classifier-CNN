import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes=None):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *(3,input_shape,input_shape))
            flat_features = self.conv_layers(dummy_input).view(1,-1).shape[1]

        self.dense = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense(x)
        return x


