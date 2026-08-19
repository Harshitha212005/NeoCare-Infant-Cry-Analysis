import os
import shutil

# ==============================
# Base Dataset Path
# ==============================
BASE_PATH = r"D:\Project\Major\Cry Datasets"

# Source datasets
dataset_paths = {
    "D1": os.path.join(BASE_PATH, "Dataset 1"),
    "D2": os.path.join(BASE_PATH, "Dataset 2"),
    "D3": os.path.join(BASE_PATH, "Dataset 3"),
}

# Destination
destination = os.path.join(BASE_PATH, "NeoCare_Dataset_V1")

# Class names
classes = [
    "Belly or Physical pain",
    "Burping",
    "Discomfort",
    "Hungry",
    "Temperature Discomfort",
    "Tired"
]

# Counters
counter = {cls: 1 for cls in classes}
total = 0

for dataset_name, dataset_path in dataset_paths.items():

    if not os.path.exists(dataset_path):
        continue

    for cls in classes:

        source_folder = os.path.join(dataset_path, cls)

        if not os.path.exists(source_folder):
            continue

        target_folder = os.path.join(destination, cls)
        os.makedirs(target_folder, exist_ok=True)

        for file in os.listdir(source_folder):

            if file.lower().endswith(".wav"):

                src = os.path.join(source_folder, file)

                filename = (
                    f"{dataset_name}_"
                    f"{cls.replace(' ','_')}_"
                    f"{counter[cls]:04d}.wav"
                )

                dst = os.path.join(target_folder, filename)

                shutil.copy2(src, dst)

                counter[cls] += 1
                total += 1

print("\n========== NeoCare Dataset V1 ==========")

for cls in classes:
    print(f"{cls:30} : {counter[cls]-1}")

print("----------------------------------------")
print(f"Total files copied : {total}")