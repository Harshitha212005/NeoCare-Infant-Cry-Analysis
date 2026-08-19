import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Dataset path
DATASET_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

# Output folder
OUTPUT_FOLDER = "docs/spectrograms"
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

    sample = random.choice(wav_files)

    file_path = os.path.join(class_folder, sample)

    y, sr = librosa.load(file_path, sr=None)

    # Compute spectrogram
    D = librosa.stft(y)
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        DB,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        cmap="magma"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram - {class_name}")

    save_path = os.path.join(
        OUTPUT_FOLDER,
        f"{class_name}_spectrogram.png"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print("\nSpectrograms generated successfully!")
print(f"Saved in: {OUTPUT_FOLDER}")