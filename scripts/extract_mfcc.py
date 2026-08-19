import os
import librosa
import numpy as np
from tqdm import tqdm

# ==============================
# Paths
# ==============================
INPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V4"
OUTPUT_PATH = r"D:\Project\Major\Cry Datasets\Features\MFCC"

SR = 16000
N_MFCC = 40
MAX_LENGTH = 300   # Number of time frames

os.makedirs(OUTPUT_PATH, exist_ok=True)

processed = 0

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

        try:
            # Load audio
            y, sr = librosa.load(input_file, sr=SR)

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=N_MFCC
            )

            # Pad or truncate
            if mfcc.shape[1] < MAX_LENGTH:
                pad_width = MAX_LENGTH - mfcc.shape[1]
                mfcc = np.pad(
                    mfcc,
                    ((0, 0), (0, pad_width)),
                    mode="constant"
                )
            else:
                mfcc = mfcc[:, :MAX_LENGTH]

            output_file = os.path.join(
                output_class,
                file.replace(".wav", ".npy")
            )

            np.save(output_file, mfcc)

            processed += 1

        except Exception as e:
            print(f"Error: {file} -> {e}")

print("\n==============================")
print("MFCC Extraction Complete")
print("==============================")
print(f"Files processed: {processed}")