import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

records = []

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):

        if file.lower().endswith(".wav"):

            path = os.path.join(class_folder, file)

            try:
                duration = librosa.get_duration(path=path)

                records.append({
                    "Class": class_name,
                    "Duration (s)": round(duration, 2)
                })

            except Exception:
                pass

df = pd.DataFrame(records)

# Save CSV
os.makedirs("docs", exist_ok=True)
df.to_csv("docs/audio_duration_report.csv", index=False)

# Print summary
print("\n========== AUDIO DURATION SUMMARY ==========\n")
print(df.groupby("Class")["Duration (s)"].describe())

# Plot
plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="Duration (s)",
    bins=30,
    kde=True,
    color="royalblue"
)

plt.title("NeoCare Dataset V3 - Audio Duration Distribution")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of Files")

plt.tight_layout()

plt.savefig(
    "docs/audio_duration_distribution.png",
    dpi=300
)

print("\nSaved:")
print("docs/audio_duration_report.csv")
print("docs/audio_duration_distribution.png")

plt.show()