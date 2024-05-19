import torch
from torch import nn

# Replace the classifier head with a regression-friendly output layer
class ViTForRegression(nn.Module):
    def __init__(self, vit_model, config, num_features):
        super(ViTForRegression, self).__init__()
        self.vit = vit_model
        self.regressor = nn.Linear(config.hidden_size, num_features)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        sequence_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        logits = self.regressor(sequence_output)
        return logits

# CNN Model
class MultiEventMusicTranscriptionCNN(nn.Module):
    def __init__(self, n_events=10):
        super(MultiEventMusicTranscriptionCNN, self).__init__()
        self.n_events = n_events
        self.output_size = n_events * 5

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, self.output_size)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), self.n_events, 5)