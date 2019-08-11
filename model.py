import torch
import torch.nn as nn
import torchvision.models as models


class Baseline(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.Conv2d(32, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.5),
            nn.Conv2d(1024, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet18(pretrained=True)
        # model = models.densenet121(pretrained=True)
        # num_ftrs = model.classifier.in_features
        # model.classifier = nn.Linear(num_ftrs, out_size)
        model = list(model.children())[:-1]
        # print(model)
        model.append(nn.Conv2d(512, out_size, 1))
        model.append(nn.Conv2d(512, out_size, 1))
        # model.append(nn.Linear(512, out_size))
        self.net = nn.Sequential(*model)
        # self.net = model

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)
