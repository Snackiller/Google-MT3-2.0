import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from collections import defaultdict
import shutil
import ast  # Add this import to the top of your script
# Dataset
class SpectrogramNoteEventDataset(Dataset):
    def __init__(self, samples, n_events=10, transform=None):
        self.samples = samples
        self.n_events = n_events
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, events, frame_start, frame_end, prefix = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = np.zeros((self.n_events, 5), dtype=np.float32)
        events = events.sort_values(by='start_time')[:self.n_events]

        for i, (_, event) in enumerate(events.iterrows()):
            if i >= self.n_events:
                break
            start_time = max(0.0, event["start_time"] - frame_start)
            end_time = min(frame_end - frame_start, event["end_time"] - frame_start)
            confidence = ast.literal_eval(event["confidence"]) if isinstance(event["confidence"], str) else event["confidence"]
            mean_confidence = np.mean(confidence) if isinstance(confidence, list) else confidence
            target[i] = [
                start_time,
                end_time,
                event["pitch"],
                event["velocity"],
                mean_confidence
            ]

        return image, target, prefix, frame_start



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


def load_data(base_dir):
    data_samples = []
    for song_dir in os.listdir(base_dir):
        song_path = os.path.join(base_dir, song_dir)
        csv_path = os.path.join(song_path, 'csv')
        spectrogram_path = os.path.join(song_path, 'spectrograms')

        for csv_file in os.listdir(csv_path):
            full_csv_path = os.path.join(csv_path, csv_file)
            events = pd.read_csv(full_csv_path)

            for img_file in os.listdir(spectrogram_path):
                img_path = os.path.join(spectrogram_path, img_file)
                match = re.match(r'.+_start([0-9.]+)_end([0-9.]+)\.png$', img_file)
                if match:
                    start_time, end_time = map(float, match.groups())
                    matched_events = events[(events["start_time"] <= end_time) & (events["end_time"] >= start_time)]
                    if not matched_events.empty:
                        data_samples.append((img_path, matched_events, start_time, end_time, song_dir))
    return data_samples


def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
def predict_and_save_csv_multi_event(model, dataloader, device, output_csv_dir, n_events=10):
    print("Starting prediction...")
    model.eval()
    os.makedirs(output_csv_dir, exist_ok=True)  # Ensures directory is clean for each run

    predictions_per_audio = {}
    processed_files_count = 0

    with torch.no_grad():
        for inputs, _, prefixes, frame_starts in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()

            for batch_index, output in enumerate(outputs):
                audio_prefix = prefixes[batch_index]
                frame_start_time = frame_starts[batch_index]

                # Debug print to confirm which prefixes are being processed
                print(f"Processing batch from dataloader with prefix: {audio_prefix}")
                processed_files_count += 1

                if audio_prefix not in predictions_per_audio:
                    predictions_per_audio[audio_prefix] = []
                    print(f"Generating predictions for {audio_prefix}")

                for i in range(n_events):
                    start_time, end_time, pitch, velocity, confidence = output[i]
                    if start_time >= end_time or confidence <= 0:
                        continue

                    predictions_per_audio[audio_prefix].append([
                        float(frame_start_time + start_time),
                        float(frame_start_time + end_time),
                        int(pitch),
                        min(max(float(velocity), 0), 127),
                        min(max(float(confidence), 0.0), 1.0)
                    ])

    output_files_count = 0
    for audio_prefix, predictions in predictions_per_audio.items():
        if predictions:  # Only create a file if there are predictions
            df = pd.DataFrame(predictions, columns=["start_time", "end_time", "pitch", "velocity", "confidence"])
            df.to_csv(os.path.join(output_csv_dir, f"{audio_prefix}_predictions.csv"), index=False)
            print(f"Predictions for {audio_prefix} saved to {output_csv_dir}")
            output_files_count += 1

    print(f"Processed {processed_files_count} input files.")
    print(f"Created {output_files_count} output files.")



 # Training, Validation, and Prediction Functions are the same as previous snippets
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets, _, _ in dataloader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

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
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    return running_loss / len(dataloader)

if __name__ == "__main__":
    # Define directories
    base_dir = "../organized_data/"  # Change this to your actual directory path
    output_csv_dir = "../CNN_Prediction_result/"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load data
    all_samples = load_data(base_dir)
    train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    test_samples, val_samples = train_test_split(test_samples, test_size=0.5, random_state=42)

    # Prepare datasets
    train_dataset = SpectrogramNoteEventDataset(train_samples, transform=transform)
    val_dataset = SpectrogramNoteEventDataset(val_samples, transform=transform)
    test_dataset = SpectrogramNoteEventDataset(test_samples, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, and optimizer
    model = MultiEventMusicTranscriptionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    model_path = "multi_event_music_transcription_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Predictions
    predict_and_save_csv_multi_event(model, test_loader, device, output_csv_dir, 10)
