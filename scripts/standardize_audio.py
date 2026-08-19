import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ==============================
# Paths
# ==============================
INPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V1"
OUTPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V2"

TARGET_SR = 16000

total = 0

for class_name in os.listdir(INPUT_PATH):

    input_class = os.path.join(INPUT_PATH, class_name)

    if not os.path.isdir(input_class):
        continue

    output_class = os.path.join(OUTPUT_PATH, class_name)
    os.makedirs(output_class, exist_ok=True)

    for file in tqdm(os.listdir(input_class), desc=class_name):

        if not file.lower().endswith(".wav"):
            continue

        input_file = os.path.join(input_class, file)
        output_file = os.path.join(output_class, file)

        try:
            # Load as mono and resample to 16 kHz
            audio, sr = librosa.load(
                input_file,
                sr=TARGET_SR,
                mono=True
            )

            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # Save as 16-bit PCM WAV
            sf.write(
                output_file,
                audio,
                TARGET_SR,
                subtype="PCM_16"
            )

            total += 1

        except Exception as e:
            print(f"Error: {file} -> {e}")

print("\n==============================")
print("Audio Standardization Complete")
print("==============================")
print(f"Files processed: {total}")