import torch
import torch.nn as nn
import torch.nn.functional as F


class Viewpoint(nn.Module):
    def __init__(self):
        super(Viewpoint, self).__init__()
        self.conv1 = nn.Conv2d(21, 64, 3, padding=1) 
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(4098, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hand_side):
        size = x.size() # (b, 21, 32, 32)
        out = self.relu(self.conv1(x)) # 1 (b, 64, 32, 32)
        out = self.relu(self.conv2(out)) # 2 (b, 64, 16, 16)
        out = self.relu(self.conv3(out)) # 3 (b, 128, 16, 16)
        out = self.relu(self.conv4(out)) # 4 (b, 128, 8, 8)
        out = self.relu(self.conv5(out)) # 5 (b, 256, 8, 8)
        out = self.relu(self.conv6(out)) # 6 (b, 256, 4, 4)

        out = torch.reshape(out, (size[0], -1)) # (b, 4096)
        out = torch.cat((out, hand_side), dim=1) # (b, 4096)

        out = self.dropout(self.relu(self.fc1(out))) # (b, 256)
        out = self.dropout(self.relu(self.fc2(out))) # (b, 128)
        
        ux = self.fc(out) # (b, 1)
        uy = self.fc(out) # (b, 1)
        uz = self.fc(out) # (b, 1)

        return torch.cat((ux, uy, uz), 1)