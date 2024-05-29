import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(base_dir, specific_song):
    song_path = os.path.join(base_dir, specific_song)
    csv_path = os.path.join(song_path, 'csv')
    spectrogram_path = os.path.join(song_path, 'spectrograms')
    song_samples = []

    # Assume there is only one CSV file per song directory
    csv_file = os.listdir(csv_path)[0]
    full_csv_path = os.path.join(csv_path, csv_file)
    events = pd.read_csv(full_csv_path)

    # Process each spectrogram image in the song directory
    for img_file in os.listdir(spectrogram_path):
        img_path = os.path.join(spectrogram_path, img_file)
        match = re.match(r'^.+_start([0-9.]+)_end([0-9.]+)\.png$', img_file)

        if match:
            start_time, end_time = map(float, match.groups())
            # Filter events that overlap with the spectrogram time interval
            matched_events = events[(events["start_time"] <= end_time) & (events["end_time"] >= start_time)]
            if not matched_events.empty:
                song_samples.append({
                    'spectrogram': img_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'matched_events': matched_events.to_dict(orient='records')
                })

    return song_samples
def display_song_samples(song_samples):
    for sample in song_samples:
        print(f"Spectrogram: {sample['spectrogram']}")
        print(f"Time Interval: {sample['start_time']} to {sample['end_time']}")
        print("Matched Note Events:")
        for event in sample['matched_events']:
            print(event)
        print("\n---------------------\n")

def split_data(song_data):
    songs = list(song_data.keys())
    train_songs, test_songs = train_test_split(songs, test_size=0.2, random_state=42)
    train_songs, val_songs = train_test_split(train_songs, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    train_samples = [item for song in train_songs for item in song_data[song]]
    val_samples = [item for song in val_songs for item in song_data[song]]
    test_samples = [item for song in test_songs for item in song_data[song]]

    return train_samples, val_samples, test_samples

def visualize_dataset_split(train_samples, val_samples, test_samples):
    # Collect all songs in each dataset
    train_songs = {sample[0].split('/')[2] for sample in train_samples}  # Adjust the index as needed
    val_songs = {sample[0].split('/')[2] for sample in val_samples}
    test_songs = {sample[0].split('/')[2] for sample in test_samples}

    # Plotting setup
    plt.figure(figsize=(10, 5))
    all_songs = sorted(list(train_songs | val_songs | test_songs))
    train_counts = [1 if song in train_songs else 0 for song in all_songs]
    val_counts = [1 if song in val_songs else 0 for song in all_songs]
    test_counts = [1 if song in test_songs else 0 for song in all_songs]

    bar_width = 0.25
    r1 = np.arange(len(all_songs))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, train_counts, color='b', width=bar_width, label='Train')
    plt.bar(r2, val_counts, color='r', width=bar_width, label='Validation')
    plt.bar(r3, test_counts, color='g', width=bar_width, label='Test')

    plt.xlabel('Songs')
    plt.xticks([r + bar_width for r in range(len(all_songs))], all_songs, rotation=90)
    plt.title('Dataset Split by Songs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def summarize_dataset(song_data, train_samples, val_samples, test_samples):
    # Total number of songs
    total_songs = len(song_data)
    print(f"Total number of songs: {total_songs}")

    # Unique songs in each dataset split
    train_songs = set(sample[0].split('/')[2] for sample in train_samples)  # Adjust the index as needed
    val_songs = set(sample[0].split('/')[2] for sample in val_samples)
    test_songs = set(sample[0].split('/')[2] for sample in test_samples)

    # Printing summary information
    print(f"Number of songs in training set: {len(train_songs)}")
    print(f"Number of songs in validation set: {len(val_songs)}")
    print(f"Number of songs in test set: {len(test_songs)}")

    # Instances per song
    for key in song_data:
        print(f"{key} - Total instances: {len(song_data[key])}")



# Main Usage
base_dir = "../organized_data/"





specific_song = "05_SS3-98-C_comp_mic"  # Example song folder name
song_samples = load_data(base_dir, specific_song)
display_song_samples(song_samples)