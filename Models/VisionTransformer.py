import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTModel, ViTConfig, ViTImageProcessor

# Dataset Class
class SpectrogramNoteEventDataset(Dataset):
    def __init__(self, spectrogram_dir, csv_dir, n_events=10, transform=None):
        self.spectrogram_dir = spectrogram_dir
        self.csv_dir = csv_dir
        self.n_events = n_events
        self.transform = transform
        self.samples = []

        for file in os.listdir(self.csv_dir):
            if file.endswith(".csv"):
                csv_path = os.path.join(self.csv_dir, file)
                spectrogram_prefix = os.path.splitext(file)[0]
                note_events = pd.read_csv(csv_path)

                for img_file in os.listdir(self.spectrogram_dir):
                    if img_file.startswith(spectrogram_prefix) and img_file.endswith(".png"):
                        try:
                            match = re.match(r'.+_start([0-9.]+)_end([0-9.]+)\.png$', img_file)
                            if not match:
                                raise ValueError(f"Error parsing start and end times from {img_file}")
                            start_time = float(match.group(1))
                            end_time = float(match.group(2))
                            img_path = os.path.join(self.spectrogram_dir, img_file)

                            events = note_events[(note_events["start_time"] < end_time) & (note_events["end_time"] > start_time)]
                            if len(events) > 0:
                                self.samples.append((img_path, events, start_time, end_time, spectrogram_prefix))
                        except ValueError as e:
                            print(e)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, events, frame_start, frame_end, prefix = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = np.zeros((self.n_events, 5), dtype=np.float32)
        for i, (_, event) in enumerate(events.iterrows()):
            if i >= self.n_events:
                break
            start_time = max(0.0, event["start_time"] - frame_start)
            end_time = min(frame_end - frame_start, event["end_time"] - frame_start)
            target[i] = [start_time, end_time, event["pitch"], event["velocity"], np.mean(eval(event["confidence"]))]
        return image, target, prefix, frame_start

# Load model configuration
config = ViTConfig.from_pretrained('google/vit-base-patch16-224')

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

# Initialize the Vision Transformer model
model = ViTModel(config)
model = ViTForRegression(model, config, num_features=50)
print(model)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Initialize the image processor
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# Dataset and DataLoader Setup
spectrogram_dir = "../BasicPitchTCN_prediction/spectrograms"
csv_dir = "../BasicPitchTCN_prediction/csv"

print("Loading dataset...")
dataset = SpectrogramNoteEventDataset(spectrogram_dir, csv_dir, n_events=10, transform=transform)
print(f"Dataset loaded with {len(dataset)} samples")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training and Validation Functions
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets, _, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 50)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets, _, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 50)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataloader)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Model Saving
model_path = "vit_music_transcription.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
