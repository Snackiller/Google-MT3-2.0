import os
import librosa
import librosa.display
import numpy as np
import jams
import pretty_midi
from interpreter import jams_to_midi  # Assuming the jams_to_midi function is correctly imported

# Directory paths
audio_dir = '../RawData/audio_mono-mic'
jams_dir = '../RawData/annotation'
output_audio_dir = '../ProcessedData/unframed_spectrograms'
output_label_dir = '../ProcessedData/midi'

# Ensure output directories exist
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Parameters for Mel spectrogram
n_fft = 2048
hop_length = 512
n_mels = 128

# Process files
for filename in os.listdir(jams_dir):
    if filename.endswith('.jams'):
        jams_path = os.path.join(jams_dir, filename)
        audio_filename = filename.replace('.jams', '_mic.wav')
        audio_path = os.path.join(audio_dir, audio_filename)

        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue

        # Load JAMS and audio
        jam = jams.load(jams_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)  # Load mono-mic audio

        # Compute Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Save the spectrogram
        spectrogram_path = os.path.join(output_audio_dir, f"{os.path.splitext(audio_filename)[0]}_mel.npy")
        np.save(spectrogram_path, log_S)

        # Convert JAMS to MIDI
        midi = jams_to_midi(jam)
        midi_filename = audio_filename.replace('.wav', '.mid')
        midi_path = os.path.join(output_label_dir, midi_filename)
        midi.write(midi_path)

        print(f"Processed and saved: {audio_filename}")

