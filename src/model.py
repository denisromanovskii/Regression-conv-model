import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self, in_channels, out):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, (3, 3), bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.linear_block = nn.Sequential(
            nn.Linear(64 * 60 * 60, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, out)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        out = self.linear_block(x)
        return out

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = ConvModel(1, 2)
model = model.to(device)
loss_func = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

