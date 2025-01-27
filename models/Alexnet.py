import torch.nn as nn
import torch

# AlexNet: Input 224x224; scale the model architecture;
# Use the sgd as optimizer, rather than adam.

class AlexNet(nn.Module):
    def __init__(self, in_channels =3, num_classes=10, **kwargs):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(x.shape)
        x = self.flatten(x)
        #x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def Alexnet(in_channels,num_classes,**kwargs):
    return AlexNet(in_channels,num_classes,**kwargs)