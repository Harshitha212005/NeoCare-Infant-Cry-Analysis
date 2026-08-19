
import os
import shutil
import librosa
import numpy as np

INPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V2"
OUTPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

os.makedirs(OUTPUT_PATH, exist_ok=True)

fingerprints = set()

unique = 0
duplicates = 0

for class_name in os.listdir(INPUT_PATH):

    input_class = os.path.join(INPUT_PATH, class_name)

    if not os.path.isdir(input_class):
        continue

    output_class = os.path.join(OUTPUT_PATH, class_name)
    os.makedirs(output_class, exist_ok=True)

    for file in os.listdir(input_class):

        if not file.endswith(".wav"):
            continue

        path = os.path.join(input_class, file)

        try:
            y, sr = librosa.load(path, sr=16000)

            # Create a simple fingerprint from MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            fingerprint = tuple(np.round(np.mean(mfcc, axis=1), 2))

            if fingerprint not in fingerprints:

                fingerprints.add(fingerprint)

                shutil.copy2(
                    path,
                    os.path.join(output_class, file)
                )

                unique += 1

            else:
                duplicates += 1

        except Exception:
            continue

print("\n========== Duplicate Removal ==========")
print(f"Unique files      : {unique}")
print(f"Duplicates removed: {duplicates}")