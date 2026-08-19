import os
import librosa
import numpy as np
from tqdm import tqdm

# ==============================
# Paths
# ==============================
INPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V4"
OUTPUT_PATH = r"D:\Project\Major\Cry Datasets\Features\LogMel"

SR = 16000
N_MELS = 128
MAX_LENGTH = 300

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
            y, sr = librosa.load(input_file, sr=SR)

            # Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=N_MELS
            )

            # Convert to Log-Mel
            log_mel = librosa.power_to_db(mel, ref=np.max)

            # Pad or truncate
            if log_mel.shape[1] < MAX_LENGTH:
                pad = MAX_LENGTH - log_mel.shape[1]
                log_mel = np.pad(
                    log_mel,
                    ((0, 0), (0, pad)),
                    mode="constant"
                )
            else:
                log_mel = log_mel[:, :MAX_LENGTH]

            output_file = os.path.join(
                output_class,
                file.replace(".wav", ".npy")
            )

            np.save(output_file, log_mel)

            processed += 1

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("\n==============================")
print("Log-Mel Extraction Complete")
print("==============================")
print(f"Files processed: {processed}")