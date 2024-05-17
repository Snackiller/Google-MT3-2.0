import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch


class SpectrogramNoteEventDataset(Dataset):
    ############################################################################
    # The SpectrogramNoteEventDataset class processes directories containing CSV
    # files with note event labels and directories containing spectrogram images.
    # These spectrograms are assumed to have been pre-generated from audio files,
    # where each spectrogram corresponds to a segment of the audio (framed Mel spectrograms).
    #############################################################################
    # Image and CSV File Association: Each image file is named in such a way that
    # it includes time stamps (start and end times) which help in matching the image
    # (spectrogram) with corresponding events in the CSV files. The script reads the
    # start and end times directly from the filenames and filters the events in the
    # CSV files to match these times. This ensures that only relevant events are
    # considered for each spectrogram frame.
    ##############################################################################
    # Data Transformation: The Mel spectrograms are loaded using the PIL
    # library, which converts them to a consistent format (RGB, even though
    # they are likely single-channel grayscale originally). They are then resized
    # and converted to tensors via a transformation pipeline. This makes them suitable
    # for processing with a convolutional neural network (CNN), which expects tensor input.
    ##############################################################################
    # Data Structure: Each data sample in the dataset includes:
    # The Mel spectrogram image tensor, ready to be used as input to the model.
    # A target array containing details about note events (start time relative to the frame,
    # end time relative to the frame, pitch, velocity, and confidence).
    # details are extracted from the filtered CSV data.


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

                            # Extract note events within the time frame
                            events = note_events[(note_events["start_time"] < end_time) & (note_events["end_time"] > start_time)]
                            if len(events) > 0:
                                self.samples.append((img_path, events, start_time, end_time))
                        except ValueError as e:
                            print(e)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, events, frame_start, frame_end = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure conversion to 3 channels
        if self.transform:
            image = self.transform(image)

        target = np.zeros((self.n_events, 5), dtype=np.float32)  # Ensure dtype is float32
        for i, (_, event) in enumerate(events.iterrows()):
            if i >= self.n_events:
                break
            start_time = max(0.0, event["start_time"] - frame_start)
            end_time = min(frame_end - frame_start, event["end_time"] - frame_start)
            target[i] = [
                start_time,
                end_time,
                event["pitch"],
                event["velocity"],
                np.mean(eval(event["confidence"]))
            ]

        return image, target

# Path setup
spectrogram_dir = "../BasicPitchTCN_prediction/spectrograms"
csv_dir = "../BasicPitchTCN_prediction/csv"

# Transformation setup
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset loading
dataset = SpectrogramNoteEventDataset(spectrogram_dir, csv_dir, n_events=10, transform=transform)
print(f"Loaded dataset with {len(dataset)} samples")

# Splitting the dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, remaining_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
val_dataset, test_dataset = torch.utils.data.random_split(remaining_dataset, [val_size, test_size])

# New directory for datasets
dataset_dir = '../READY_datasets'
os.makedirs(dataset_dir, exist_ok=True)

# Saving datasets
torch.save(train_dataset, os.path.join(dataset_dir, 'train_dataset.pth'))
torch.save(val_dataset, os.path.join(dataset_dir, 'val_dataset.pth'))
torch.save(test_dataset, os.path.join(dataset_dir, 'test_dataset.pth'))

print(f"Datasets saved in {dataset_dir}: train_dataset.pth, val_dataset.pth, and test_dataset.pth")
