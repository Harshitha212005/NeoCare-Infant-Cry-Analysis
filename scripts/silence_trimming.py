import os
import librosa
import soundfile as sf
from tqdm import tqdm

INPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"
OUTPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V4"

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
        output_file = os.path.join(output_class, file)

        try:
            # Load standardized audio
            y, sr = librosa.load(input_file, sr=16000)

            # Remove leading and trailing silence
            y_trimmed, _ = librosa.effects.trim(
                y,
                top_db=20
            )

            # Save trimmed audio
            sf.write(
                output_file,
                y_trimmed,
                sr,
                subtype="PCM_16"
            )

            processed += 1

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("\n==============================")
print("Silence Trimming Complete")
print("==============================")
print(f"Files processed: {processed}")