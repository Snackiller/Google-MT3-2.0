
import jams

def load_jams_file(filepath):
    return jams.load(filepath)

def print_beat_and_chord_annotations(jam):
    beats = jam.search(namespace='beat')
    chords = jam.search(namespace='chord')

    print("Beats:")
    for beat in beats[0]:
        print(f"Time: {beat.time}, Beat: {beat.value}")

    print("\nChords:")
    for chord in chords[0]:
        print(f"Time: {chord.time}, Duration: {chord.duration}, Chord: {chord.value}")

# Usage
filepath = '../RawData/annotation/00_BN1-129-Eb_comp.jams'
jam = load_jams_file(filepath)
print_beat_and_chord_annotations(jam)
