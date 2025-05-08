import torch.nn as nn
import torch
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        dummy_input = torch.zeros(1, 1, 40, 300)  # shape: [batch, channel, freq, time]
        with torch.no_grad():
            dummy_out = self.conv(dummy_input)
            self.flattened_size = dummy_out.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, F, T]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x