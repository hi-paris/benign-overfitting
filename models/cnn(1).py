import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self,width):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, width, (3, 3))
        self.conv2 = nn.Conv2d(width, width, (3, 3))
        self.conv3 = nn.Conv2d(width, width, (3, 3))
        self.fc = nn.Linear(26*26*width, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x