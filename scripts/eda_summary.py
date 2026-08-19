import os
import librosa
import pandas as pd

# ==========================================
# Dataset Path
# ==========================================
DATASET_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

records = []

for class_name in sorted(os.listdir(DATASET_PATH)):

    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):

        if file.lower().endswith(".wav"):

            file_path = os.path.join(class_folder, file)

            try:
                y, sr = librosa.load(file_path, sr=None, mono=False)

                duration = librosa.get_duration(y=y, sr=sr)
                channels = 1 if y.ndim == 1 else y.shape[0]

                records.append({
                    "Class": class_name,
                    "File": file,
                    "Duration (s)": round(duration, 2),
                    "Sample Rate": sr,
                    "Channels": channels
                })

            except Exception:
                print(f"Error reading: {file}")

# Create DataFrame
df = pd.DataFrame(records)

# Create docs folder if it doesn't exist
os.makedirs("docs", exist_ok=True)

# Save detailed report
df.to_csv("docs/dataset_summary.csv", index=False)

# Print overall statistics
print("\n========== DATASET SUMMARY ==========\n")

print(f"Total Classes       : {df['Class'].nunique()}")
print(f"Total Audio Files   : {len(df)}")
print(f"Average Duration    : {df['Duration (s)'].mean():.2f} sec")
print(f"Minimum Duration    : {df['Duration (s)'].min():.2f} sec")
print(f"Maximum Duration    : {df['Duration (s)'].max():.2f} sec")

print("\nFiles per Class")
print(df["Class"].value_counts())

print("\nSample Rate Distribution")
print(df["Sample Rate"].value_counts())

print("\nChannel Distribution")
print(df["Channels"].value_counts())

# Save summary text
with open("docs/dataset_summary_report.txt", "w") as f:

    f.write("NeoCare Dataset Summary\n")
    f.write("=======================\n\n")

    f.write(f"Total Classes      : {df['Class'].nunique()}\n")
    f.write(f"Total Audio Files  : {len(df)}\n")
    f.write(f"Average Duration   : {df['Duration (s)'].mean():.2f} sec\n")
    f.write(f"Minimum Duration   : {df['Duration (s)'].min():.2f} sec\n")
    f.write(f"Maximum Duration   : {df['Duration (s)'].max():.2f} sec\n\n")

    f.write("Files per Class\n")
    f.write(str(df["Class"].value_counts()))
    f.write("\n\nSample Rate Distribution\n")
    f.write(str(df["Sample Rate"].value_counts()))
    f.write("\n\nChannel Distribution\n")
    f.write(str(df["Channels"].value_counts()))

print("\nReports saved successfully!")
print("docs/dataset_summary.csv")
print("docs/dataset_summary_report.txt")