import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from time import time
from mtutil import SpectrogramNoteEventDataset
from mtconfig import SEED, SPECTROGRAM_DIR, CSV_DIR
from mtmodels import MultiEventMusicTranscriptionCNN

# from basic_pitch.inference import predict, Model
# from basic_pitch import ICASSP_2022_MODEL_PATH

# Set random seed for reproducibility
torch.manual_seed(SEED)

# Prediction and CSV Saving Function
def predict_and_save_csv_multi_event(model, dataloader, device, output_csv_dir, n_events=10):
    model.eval()
    os.makedirs(output_csv_dir, exist_ok=True)

    predictions_per_audio = {}

    with torch.no_grad():
        for inputs, _, prefix, frame_start in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()

            for batch_index, output in enumerate(outputs):
                audio_prefix = prefix[batch_index]
                frame_start_time = frame_start[batch_index]

                if audio_prefix not in predictions_per_audio:
                    predictions_per_audio[audio_prefix] = []

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

    # Save each prediction to a separate CSV file
    for audio_prefix, predictions in predictions_per_audio.items():
        df = pd.DataFrame(predictions, columns=["start_time", "end_time", "pitch", "velocity", "confidence"])
        df.to_csv(os.path.join(output_csv_dir, f"{audio_prefix}_predictions.csv"), index=False)

    print(f"Predictions saved to {output_csv_dir}")
# Training Functions
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

output_csv_dir = "../CNN_Prediction_result/"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

print("Loading dataset...")
dataset = SpectrogramNoteEventDataset(SPECTROGRAM_DIR, CSV_DIR, n_events=10, transform=transform)
print(f"Dataset loaded with {len(dataset)} samples")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = MultiEventMusicTranscriptionCNN(n_events=10).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
start_time = time()
for epoch in range(num_epochs):
    epoch_start_time = time()
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    print(f"Time taken in thie epoch: {time() - epoch_start_time:.2f}s") 
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
print(f"Total time taken: {time() - start_time:.2f}s")

# Save the Model
model_path = "multi_event_music_transcription_cnn.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Predict and Save Predictions as CSV
predict_and_save_csv_multi_event(model, val_loader, device, output_csv_dir, n_events=10)

#
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import transforms
# import sys
# sys.path.append('../Preprocessor/')  # Add the directory, not the file
# from DatasetBuilding_Saving import SpectrogramNoteEventDataset  # Import class
# import pandas as pd
# # CNN Model Definition
# class MultiEventMusicTranscriptionCNN(nn.Module):
#     def __init__(self, n_events=10):
#         super(MultiEventMusicTranscriptionCNN, self).__init__()
#         self.n_events = n_events
#         self.output_size = n_events * 5
#
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 8 * 8, 128)
#         self.fc2 = nn.Linear(128, self.output_size)
#
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.maxpool(x)
#
#         x = self.relu(self.conv2(x))
#         x = self.maxpool(x)
#
#         x = self.relu(self.conv3(x))
#         x = self.maxpool(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x.view(x.size(0), self.n_events, 5)
#
# # Function to load datasets
# def load_tensor_dataset(file_path):
#     data = torch.load(file_path)
#     return data
# # Function to predict and save CSV for multiple events
# def predict_and_save_csv_multi_event(model, dataloader, device, output_csv_dir, n_events=10):
#     model.eval()
#     predictions_per_audio = {}
#
#     with torch.no_grad():
#         for inputs, targets in dataloader:
#             inputs = inputs.to(device)
#             outputs = model(inputs).cpu().numpy()
#
#             for batch_index, output in enumerate(outputs):
#                 audio_prefix = targets[batch_index][0]  # Assuming first item in targets is the prefix
#                 frame_start_time = targets[batch_index][1]  # Assuming second item is frame start time
#
#                 if audio_prefix not in predictions_per_audio:
#                     predictions_per_audio[audio_prefix] = []
#
#                 for i in range(n_events):
#                     start_time, end_time, pitch, velocity, confidence = output[i]
#                     if start_time >= end_time or confidence <= 0:
#                         continue
#
#                     predictions_per_audio[audio_prefix].append([
#                         float(frame_start_time + start_time),
#                         float(frame_start_time + end_time),
#                         int(pitch),
#                         min(max(int(velocity), 0), 127),
#                         min(max(float(confidence), 0.0), 1.0)
#                     ])
#
#     # Save each prediction to a separate CSV file
#     for audio_prefix, predictions in predictions_per_audio.items():
#         df = pd.DataFrame(predictions, columns=["start_time", "end_time", "pitch", "velocity", "confidence"])
#         df.to_csv(os.path.join(output_csv_dir, f"{audio_prefix}_predictions.csv"), index=False)
#
#     print(f"Predictions saved to {output_csv_dir}")
#
#
# # Paths
# # Define the directory for data and output
# data_dir = '../READY_datasets'
# output_csv_dir = '../CNN_Prediction_result'
# os.makedirs(output_csv_dir, exist_ok=True)
#
# # Loading datasets
# train_dataset = load_tensor_dataset(os.path.join(data_dir, 'train_dataset.pth'))
# val_dataset = load_tensor_dataset(os.path.join(data_dir, 'val_dataset.pth'))
# test_dataset = load_tensor_dataset(os.path.join(data_dir, 'test_dataset.pth'))
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # Setup device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Model instantiation
# model = MultiEventMusicTranscriptionCNN(n_events=10).to(device)
#
# # Loss and Optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training Loop
# def train(model, dataloader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     for inputs, targets in dataloader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     return running_loss / len(dataloader)
#
# # Validation Loop
# def validate(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item()
#     return running_loss / len(dataloader)
#
# num_epochs = 10
# for epoch in range(num_epochs):
#     train_loss = train(model, train_loader, criterion, optimizer, device)
#     val_loss = validate(model, val_loader, criterion, device)
#     print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
#
# # Save Model
# model_path = 'multi_event_music_transcription_cnn.pth'
# torch.save(model.state_dict(), model_path)
# print(f'Model saved to {model_path}')

