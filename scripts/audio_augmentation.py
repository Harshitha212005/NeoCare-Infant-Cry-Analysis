import os
import librosa
import soundfile as sf
import numpy as np

# =====================================
# Paths
# =====================================

INPUT_DATASET = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V4"

OUTPUT_DATASET = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V5"

os.makedirs(OUTPUT_DATASET, exist_ok=True)

TARGET_SR = 16000

# =====================================
# Augmentation Functions
# =====================================

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def time_shift(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)


def pitch_shift(audio, sr):
    return librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=2
    )


def time_stretch(audio):
    return librosa.effects.time_stretch(
        audio,
        rate=1.1
    )

# =====================================
# Process Dataset
# =====================================

for class_name in os.listdir(INPUT_DATASET):

    input_folder = os.path.join(INPUT_DATASET, class_name)

    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(OUTPUT_DATASET, class_name)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing {class_name}")

    for file in os.listdir(input_folder):

        if not file.endswith(".wav"):
            continue

        path = os.path.join(input_folder, file)

        audio, sr = librosa.load(
            path,
            sr=TARGET_SR
        )

        # Save original
        sf.write(
            os.path.join(output_folder, file),
            audio,
            TARGET_SR
        )

        # Noise
        sf.write(
            os.path.join(output_folder, f"noise_{file}"),
            add_noise(audio),
            TARGET_SR
        )

        # Shift
        sf.write(
            os.path.join(output_folder, f"shift_{file}"),
            time_shift(audio),
            TARGET_SR
        )

        # Pitch
        sf.write(
            os.path.join(output_folder, f"pitch_{file}"),
            pitch_shift(audio, TARGET_SR),
            TARGET_SR
        )

        # Stretch
        stretched = time_stretch(audio)

        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(
                stretched,
                (0, len(audio)-len(stretched))
            )

        sf.write(
            os.path.join(output_folder, f"stretch_{file}"),
            stretched,
            TARGET_SR
        )

print("\n================================")
print("Audio Augmentation Completed!")
print("================================")