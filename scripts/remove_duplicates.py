import os
import hashlib
import shutil

INPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V2"
OUTPUT_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

seen_hashes = set()
duplicates = 0
copied = 0

for class_name in os.listdir(INPUT_PATH):

    input_class = os.path.join(INPUT_PATH, class_name)

    if not os.path.isdir(input_class):
        continue

    output_class = os.path.join(OUTPUT_PATH, class_name)
    os.makedirs(output_class, exist_ok=True)

    for file in os.listdir(input_class):

        if not file.lower().endswith(".wav"):
            continue

        input_file = os.path.join(input_class, file)

        with open(input_file, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash not in seen_hashes:
            seen_hashes.add(file_hash)
            shutil.copy2(input_file, os.path.join(output_class, file))
            copied += 1
        else:
            duplicates += 1

print("\n========== Duplicate Removal ==========")
print(f"Unique files      : {copied}")
print(f"Duplicates removed: {duplicates}")