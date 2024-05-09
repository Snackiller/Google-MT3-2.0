import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
# Paths
input_audio_dir = "../RawData/audio_mono-mic"
output_base_dir = "../BasicPitchTCN_prediction"
midi_dir = os.path.join(output_base_dir, "midi")
npz_dir = os.path.join(output_base_dir, "npz")
csv_dir = os.path.join(output_base_dir, "csv")
spectrogram_dir = os.path.join(output_base_dir, "spectrograms")

# Create output directories
os.makedirs(midi_dir, exist_ok=True)
os.makedirs(npz_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(spectrogram_dir, exist_ok=True)

# Load the TCN model
basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)


# Function to save MIDI file
def save_midi(midi_data, output_path):
    midi_data.write(output_path)


# Function to save model outputs as an NPZ file
def save_model_outputs(model_output, output_path):
    np.savez_compressed(output_path, **model_output)


# Function to save note events as a CSV file
def save_note_events(note_events, output_path):
    columns = ["start_time", "end_time", "pitch", "velocity", "confidence"]
    df = pd.DataFrame(note_events, columns=columns)
    df.to_csv(output_path, index=False)


# Function to generate and save Mel spectrograms
def save_spectrogram_frame(y_segment, output_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Pad the audio segment with zeros if its length is shorter than n_fft
    if len(y_segment) < n_fft:
        padding = n_fft - len(y_segment)
        y_segment = np.pad(y_segment, (0, padding), mode='constant')

    S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Function to extract and save spectrograms based on note events
# length of the frame is 2.0 seconds and hop length is 0.5 seconds
def extract_spectrograms(audio_file, csv_file, spectrogram_dir, frame_length=2.0, hop_length=0.5):
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    note_events = pd.read_csv(csv_file)

    audio_duration = librosa.get_duration(y=y, sr=sr)
    frame_length_samples = int(frame_length * sr)
    hop_length_samples = int(hop_length * sr)

    for frame_start in np.arange(0, audio_duration, hop_length):
        frame_end = min(frame_start + frame_length, audio_duration)
        start_sample = int(frame_start * sr)
        end_sample = int(frame_end * sr)
        y_segment = y[start_sample:end_sample]

        # Save the spectrogram frame
        spectrogram_path = os.path.join(spectrogram_dir,
                                        f"{os.path.splitext(os.path.basename(audio_file))[0]}_start{frame_start:.4f}_end{frame_end:.4f}.png")
        save_spectrogram_frame(y_segment, spectrogram_path)


# Generate predictions and save each type of output
audio_files = [os.path.join(input_audio_dir, file) for file in os.listdir(input_audio_dir) if file.endswith(".wav")]

for audio_file in audio_files:
    # Get the base file name
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    # Run prediction
    model_output, midi_data, note_events = predict(audio_file, basic_pitch_model)

    # Save MIDI
    save_midi(midi_data, os.path.join(midi_dir, f"{base_name}.mid"))

    # Save model outputs as NPZ
    save_model_outputs(model_output, os.path.join(npz_dir, f"{base_name}.npz"))

    # Save note events as CSV
    csv_path = os.path.join(csv_dir, f"{base_name}.csv")
    save_note_events(note_events, csv_path)

    # Extract and save spectrograms
    extract_spectrograms(audio_file, csv_path, spectrogram_dir)

