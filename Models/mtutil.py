import re
import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

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

                            # Extract note events within the time frame
                            events = note_events[(note_events["start_time"] < end_time) & (note_events["end_time"] > start_time)]
                            if len(events) > 0:
                                self.samples.append((img_path, events, start_time, end_time, spectrogram_prefix))
                        except ValueError as e:
                            print(e)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, events, frame_start, frame_end, prefix = self.samples[idx]
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

        return image, target, prefix, frame_start