import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# Load the Dataset
df = pd.read_csv('crop_recommendation.csv')
print("Dataset loaded successfully.\n")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# Label Distribution
plt.figure(figsize=(12, 8))
sns.countplot(y='label', data=df, order=df['label'].value_counts().index)
plt.title('Crop Distribution')
plt.xlabel('Count')
plt.ylabel('Crop Type')
plt.tight_layout()
plt.savefig('crop_distribution.png')
plt.close()


# Only select numeric columns
numeric_df = df.select_dtypes(include='number')


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()


# Box Plot for Each Feature vs. Crop
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

for feature in features:
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='label', y=feature, data=df)
    plt.xticks(rotation=90)
    plt.title(f'{feature} vs Crop')
    plt.tight_layout()
    plt.savefig(f'{feature}_vs_crop.png')
    plt.close()