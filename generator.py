import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.main(z)
        return out.view(z.size(0), 1, 28, 28)
