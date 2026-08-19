import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

# Output folder
OUTPUT_FOLDER = "docs/mfcc"
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

    sample_file = random.choice(wav_files)
    file_path = os.path.join(class_folder, sample_file)

    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        cmap="viridis"
    )

    plt.colorbar()
    plt.title(f"MFCC - {class_name}")

    save_path = os.path.join(
        OUTPUT_FOLDER,
        f"{class_name}_mfcc.png"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print("\nMFCC visualizations generated successfully!")
print(f"Saved in: {OUTPUT_FOLDER}")