import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import re
def organize_data(spectrogram_dir, csv_dir, base_output_dir):
    # Create base directories if they don't exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Organize files into song-specific folders
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            prefix = os.path.splitext(csv_file)[0]
            song_dir = os.path.join(base_output_dir, prefix)
            spectrogram_subdir = os.path.join(song_dir, 'spectrograms')
            csv_subdir = os.path.join(song_dir, 'csv')

            # Create directories
            os.makedirs(spectrogram_subdir, exist_ok=True)
            os.makedirs(csv_subdir, exist_ok=True)

            # Move CSV file
            copyfile(os.path.join(csv_dir, csv_file), os.path.join(csv_subdir, csv_file))

            # Move corresponding spectrograms
            for img_file in os.listdir(spectrogram_dir):
                if img_file.startswith(prefix):
                    copyfile(os.path.join(spectrogram_dir, img_file), os.path.join(spectrogram_subdir, img_file))

def load_data(base_dir):
    data_samples = []
    for song_dir in os.listdir(base_dir):
        spectrogram_dir = os.path.join(base_dir, song_dir, 'spectrograms')
        csv_dir = os.path.join(base_dir, song_dir, 'csv')
        for csv_file in os.listdir(csv_dir):
            csv_path = os.path.join(csv_dir, csv_file)
            events = pd.read_csv(csv_path)
            for img_file in os.listdir(spectrogram_dir):
                img_path = os.path.join(spectrogram_dir, img_file)
                match = re.match(r'.+_start([0-9.]+)_end([0-9.]+)\.png$', img_file)
                if match:
                    start_time, end_time = map(float, match.groups())
                    matched_events = events[(events["start_time"] < end_time) & (events["end_time"] > start_time)]
                    if not matched_events.empty:
                        data_samples.append((img_path, matched_events))
    return data_samples

# Define paths
spectrogram_dir = "../BasicPitchTCN_prediction/spectrograms"
csv_dir = "../BasicPitchTCN_prediction/csv"
organized_data_dir = "../organized_data"

# Organize data
organize_data(spectrogram_dir, csv_dir, organized_data_dir)

# Load organized data
data_samples = load_data(organized_data_dir)

# Example usage of the split
train_samples, test_samples = train_test_split(data_samples, test_size=0.2, random_state=42)
