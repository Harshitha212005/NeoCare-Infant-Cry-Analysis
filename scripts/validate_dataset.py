import os
import librosa
import pandas as pd
from tqdm import tqdm

# Path to your merged dataset
BASE_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V1"

records = []

for class_name in os.listdir(BASE_PATH):

    class_folder = os.path.join(BASE_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    for file in tqdm(os.listdir(class_folder), desc=class_name):

        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(class_folder, file)

        try:
            # Load audio without changing its sample rate
            y, sr = librosa.load(file_path, sr=None, mono=False)

            # Determine mono/stereo
            channels = 1 if y.ndim == 1 else y.shape[0]

            # Duration
            duration = librosa.get_duration(y=y, sr=sr)

            records.append({
                "File": file,
                "Class": class_name,
                "Sample Rate": sr,
                "Channels": channels,
                "Duration (seconds)": round(duration, 2),
                "Status": "OK"
            })

        except Exception as e:

            records.append({
                "File": file,
                "Class": class_name,
                "Sample Rate": "",
                "Channels": "",
                "Duration (seconds)": "",
                "Status": f"ERROR: {e}"
            })

# Save report
df = pd.DataFrame(records)
df.to_csv("dataset_validation_report.csv", index=False)

print("\n========== VALIDATION COMPLETE ==========")
print(df["Status"].value_counts())
print("\nReport saved as: dataset_validation_report.csv")