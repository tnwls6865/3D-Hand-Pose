import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5_1 = nn.Conv2d(128, 512, 1)
        self.conv5_2 = nn.Conv2d(512, 21, 1)
        self.pool = nn.MaxPool2d(2, 2)     

        self.conv6_1 = nn.Conv2d(149, 128, 7, padding=3)
        self.conv6_2 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_3 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_4 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_5 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_6 = nn.Conv2d(128, 128, 1)
        self.conv6_7 = nn.Conv2d(128, 21, 1)

        self.conv7_1 = nn.Conv2d(149, 128, 7, padding=3)
        self.conv7_2 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_3 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_4 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_5 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_6 = nn.Conv2d(128, 128, 1)
        self.conv7_7 = nn.Conv2d(128, 21, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        out = self.relu(self.conv1_1(x)) 
        out = self.relu(self.conv1_2(out)) 
        out = self.pool(out) 

        out = self.relu(self.conv2_1(out)) 
        out = self.relu(self.conv2_2(out)) 
        out = self.pool(out) 

        out = self.relu(self.conv3_1(out)) 
        out = self.relu(self.conv3_2(out)) 
        out = self.relu(self.conv3_3(out)) 
        out = self.relu(self.conv3_4(out)) 
        out = self.pool(out) 
        
        out = self.relu(self.conv4_1(out)) 
        out = self.relu(self.conv4_2(out)) 
        out = self.relu(self.conv4_3(out)) 
        out = self.relu(self.conv4_4(out)) 
        out = self.relu(self.conv4_5(out)) 
        out = self.relu(self.conv4_6(out)) 
        out2 = self.relu(self.conv4_7(out))
        out = self.relu(self.conv5_1(out2))
        scoremap = self.conv5_2(out)

        out = torch.cat([scoremap, out2], dim=1) 
        out = self.relu(self.conv6_1(out)) 
        out = self.relu(self.conv6_2(out)) 
        out = self.relu(self.conv6_3(out)) 
        out = self.relu(self.conv6_4(out)) 
        out = self.relu(self.conv6_5(out)) 
        out = self.relu(self.conv6_6(out)) 
        scoremap = self.conv6_7(out) 

        out = torch.cat([scoremap, out2], dim=1) 
        out = self.relu(self.conv7_1(out)) 
        out = self.relu(self.conv7_2(out)) 
        out = self.relu(self.conv7_3(out)) 
        out = self.relu(self.conv7_4(out)) 
        out = self.relu(self.conv7_5(out)) 
        out = self.relu(self.conv7_6(out)) 
        out = self.conv7_7(out) 

        return out
        


        

        
        

