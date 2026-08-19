import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

sample_rates = []

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):

        if file.lower().endswith(".wav"):

            path = os.path.join(class_folder, file)

            try:
                _, sr = librosa.load(path, sr=None)

                sample_rates.append({
                    "Class": class_name,
                    "Sample Rate": sr
                })

            except Exception:
                pass

df = pd.DataFrame(sample_rates)

os.makedirs("docs", exist_ok=True)
df.to_csv("docs/sample_rate_report.csv", index=False)

print("\n========== SAMPLE RATE SUMMARY ==========\n")
print(df["Sample Rate"].value_counts())

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Sample Rate",
    hue="Sample Rate",
    palette="viridis",
    legend=False
)

plt.title("Sample Rate Distribution")
plt.tight_layout()

plt.savefig("docs/sample_rate_distribution.png", dpi=300)

print("\nSaved:")
print("docs/sample_rate_report.csv")
print("docs/sample_rate_distribution.png")

plt.show()