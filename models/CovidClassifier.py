import torch
import torch.nn as nn


class CovidClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CovidClassifier, self).__init__()
        self.model = nn.Sequential(
            # Input Layer
            nn.Linear(in_features=input_dim, out_features=32),
            nn.ReLU(),

            # Hidden Layers
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),

            # Output Layer
            nn.Linear(in_features=16, out_features=output_dim)
        )

    def forward(self, input_data):
        return self.model(input_data)




