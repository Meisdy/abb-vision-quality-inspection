import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 2, stride=2),
            nn.Sigmoid(),
        )

        # Optional attention network
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if self.use_attention:
            attention_mask = self.attention(x)
            return decoded, attention_mask
        else:
            return decoded
