import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset path
DATASET_PATH = r"D:\Project\Major\Cry Datasets\NeoCare_Dataset_V3"

# Count files in each class
class_counts = {}

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_folder = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_folder):
        count = len([
            f for f in os.listdir(class_folder)
            if f.lower().endswith(".wav")
        ])
        class_counts[class_name] = count

# Create DataFrame
df = pd.DataFrame(
    class_counts.items(),
    columns=["Class", "Number of Files"]
)

print("\n========== CLASS DISTRIBUTION ==========\n")
print(df)

# Save CSV
os.makedirs("docs", exist_ok=True)
df.to_csv("docs/class_distribution.csv", index=False)

# Plot
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")

ax = sns.barplot(
    data=df,
    x="Class",
    y="Number of Files",
    hue="Class",
    palette="viridis",
    legend=False
)

# Add values above bars
for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 5,
        int(bar.get_height()),
        ha="center",
        fontsize=10
    )

plt.title("NeoCare Dataset V3 - Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Audio Files")
plt.xticks(rotation=15)
plt.tight_layout()

# Save figure
plt.savefig("docs/class_distribution.png", dpi=300)

print("\nSaved:")
print("docs/class_distribution.csv")
print("docs/class_distribution.png")

plt.show()