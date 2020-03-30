import torch
import torch.nn as nn
import torch.nn.functional as F

class PosePrior(nn.Module):
    def __init__(self):
        super(PosePrior, self).__init__()
        self.conv1 = nn.Conv2d(21, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(2050, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 63)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hand_side):
        size = x.size()
        out = self.relu(self.conv1(x)) # 1 (b, 32, 32, 32)
        out = self.relu(self.conv2(out)) # 2 (b, 32, 16, 16)
        out = self.relu(self.conv3(out)) # 3 (b, 64, 16, 16)
        out = self.relu(self.conv4(out)) # 4 (b, 64, 8, 8)
        out = self.relu(self.conv5(out)) # 5 (b, 128, 8, 8)
        out = self.relu(self.conv6(out)) # 6 (b, 128, 4, 4)

        out = torch.reshape(out, (size[0], -1)) 
        out = torch.cat((out, hand_side), dim=1) 

        out = self.dropout(self.relu(self.fc1(out))) # 8 (b, 512)
        out = self.dropout(self.relu(self.fc2(out))) # 9 (b, 512)
        out = self.fc3(out) # 10 (b, 63)

        return torch.reshape(out, (size[0], 21, 3))