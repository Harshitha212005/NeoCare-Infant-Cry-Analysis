import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

# Save plots here
OUTPUT_FOLDER = "docs/waveforms"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for class_name in sorted(os.listdir(DATASET_PATH)):

    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    wav_files = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith(".wav")
    ]

    if not wav_files:
        continue

    # Pick one random file
    sample = random.choice(wav_files)

    file_path = os.path.join(class_folder, sample)

    y, sr = librosa.load(file_path, sr=None)

    plt.figure(figsize=(12,4))

    librosa.display.waveshow(y, sr=sr)

    plt.title(f"Waveform - {class_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_FOLDER,
        f"{class_name}_waveform.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

print("\nWaveform plots saved successfully!")
print(f"Location: {OUTPUT_FOLDER}")