import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self,width):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, width,kernel_size=3,stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(width, width*2,kernel_size=3,stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(width*2, width*4,kernel_size=3,stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(width*4, width*8,kernel_size=3,stride=1, padding=1, bias=True)

        self.batch_norm1= nn.BatchNorm2d(width)
        self.batch_norm2= nn.BatchNorm2d(width*2)
        self.batch_norm3= nn.BatchNorm2d(width*4)
        self.batch_norm4= nn.BatchNorm2d(width*8)

        self.max_pool2= nn.MaxPool2d(2)
        self.max_pool4= nn.MaxPool2d(4)

        self.fc = nn.Linear(width*8, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = x.relu()

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = x.relu()
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = x.relu()
        x = self.max_pool2(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = x.relu()
        x = self.max_pool2(x)

        x = self.max_pool4(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x