import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/obesity_cleaned.csv")

os.makedirs("output/plots", exist_ok=True)

# plot obesity class distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='nobeyesdad', order=sorted(df['nobeyesdad'].unique()))
plt.xticks(rotation=45)
plt.title("Obesity Class Distribution")
plt.tight_layout()
plt.savefig("output/plots/obesity_class_distribution.png")
plt.close()

# plot veggie consumption
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='nobeyesdad', y='fcvc')
plt.xticks(rotation=45)
plt.title("Vegetable Consumption (FCVC) vs Obesity Class")
plt.tight_layout()
plt.savefig("output/plots/fcvc_vs_obesity.png")
plt.close()

# plot screen time
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='nobeyesdad', y='tue')
plt.xticks(rotation=45)
plt.title("Screen Time (TUE) vs Obesity Class")
plt.tight_layout()
plt.savefig("output/plots/tue_vs_obesity.png")
plt.close()

# plot transport method
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='mtrans', hue='nobeyesdad')
plt.xticks(rotation=45)
plt.title("Transportation Method vs Obesity Class")
plt.tight_layout()
plt.savefig("output/plots/transport_vs_obesity.png")
plt.close()

# plot correlation heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/plots/correlation_heatmap.png")
plt.close()

# done
print("###################### EXPLORATORY ANALYSIS COMPLETE ######################")
