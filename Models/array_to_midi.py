import os
import csv
import pretty_midi
import glob

# Directory containing the CSV files
csv_file_path = '../BasicPitchTCN_prediction/csv/'
# Directory to save the output MIDI files
output_path = '../Predicted_MIDI_files/'

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_file_path, '*.csv'))

# Function to convert a single CSV file to a MIDI file
def convert_csv_to_midi(csv_file_path, midi_file_path):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an Instrument instance for an Acoustic Guitar (nylon) instrument (program number 24)
    guitar = pretty_midi.Instrument(program=24)
    
    # Open and read the CSV file
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row
        next(reader)
        
        for row in reader:
            start_time, end_time, pitch, velocity, _ = row[:5]
            start_time = float(start_time)
            end_time = float(end_time)
            pitch = int(pitch)
            velocity = int(float(velocity) * 127)
            
            # Ensure pitch and velocity are within the MIDI range
            pitch = max(0, min(127, pitch))
            velocity = max(0, min(127, velocity))
            
            # Create a Note instance, starting at `start_time` seconds and ending at `end_time` seconds
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
            
            # Add the note to the instrument
            guitar.notes.append(note)
    
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(guitar)
    
    # Write out the MIDI data
    midi.write(midi_file_path)
    print(f"MIDI file has been saved to {midi_file_path}")

# Traverse through each CSV file and convert to MIDI
for csv_file in csv_files:
    # Extract the base name of the CSV file to use for the MIDI file name
    base_name = os.path.basename(csv_file)
    midi_file_name = os.path.splitext(base_name)[0] + '.mid'
    midi_file_path = os.path.join(output_path, midi_file_name)
    
    # Convert the CSV to a MIDI file
    convert_csv_to_midi(csv_file, midi_file_path)
