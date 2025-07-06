import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")


# Load the Dataset
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, '../Dataset/crop_recommendation.csv')
visual_path = os.path.join(base_dir, '../Visualisations')
os.makedirs(visual_path, exist_ok=True)

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully.\n")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# Plot: Crop Distribution
plt.figure(figsize=(12, 8))
sns.countplot(y='label', data=df, order=df['label'].value_counts().index)
plt.title('Crop Distribution')
plt.xlabel('Count')
plt.ylabel('Crop Type')
plt.tight_layout()
plt.savefig(os.path.join(visual_path, 'crop_distribution.png'))
plt.close()


# Correlation Heatmap
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(visual_path, 'correlation_heatmap.png'))
plt.close()


# Box Plot for Each Feature vs. Crop
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for feature in features:
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='label', y=feature, data=df)
    plt.xticks(rotation=90)
    plt.title(f'{feature} vs Crop')
    plt.tight_layout()
    plt.savefig(os.path.join(visual_path, f'{feature}_vs_crop.png'))
    plt.close()


# Pair Plot
sns.pairplot(df, hue='label', corner=True)
plt.tight_layout()
plt.savefig(os.path.join(visual_path, 'pairplot.png'))
plt.close()