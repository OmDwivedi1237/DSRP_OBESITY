import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from textwrap import fill

sns.set_theme(style="whitegrid", context="talk")

df = pd.read_csv("data/obesity_features.csv")

FINAL_DIR = "output/final_visuals"
os.makedirs(FINAL_DIR, exist_ok=True)

def save_fig(name: str):
    path = os.path.join(FINAL_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# plot obesity class distribution
plt.figure(figsize=(10, 6))
if 'nobeyesdad' in df.columns:
    order = df.groupby(['target','nobeyesdad']).size().reset_index().sort_values('target')['nobeyesdad']
    sns.countplot(data=df, x='nobeyesdad', order=order.unique(), palette='viridis')
    plt.xlabel("Obesity Class")
    plt.title("Distribution of Obesity Classes")
else:
    sns.countplot(data=df, x='target', palette='viridis')
    plt.xlabel("Obesity Class (Encoded)")
    plt.title("Distribution of Obesity Classes (Encoded)")
plt.ylabel("Count")
save_fig("obesity_class_distribution.png")

# plot screen time vs obesity class
plt.figure(figsize=(11, 6))
sns.boxplot(data=df, x='target', y='tue', showfliers=False, palette='Set2')
sns.stripplot(data=df, x='target', y='tue', color='k', alpha=0.25, size=2)
plt.title("Daily Screen Time (TUE) by Obesity Class")
plt.xlabel("Obesity Class (Encoded)")
plt.ylabel("Screen Time (hours)")
save_fig("tue_vs_obesity.png")

# plot unhealthy cluster by transport
if {'unhealthy_cluster','active_transport'}.issubset(df.columns):
    ct = (df.groupby(['active_transport','unhealthy_cluster']).size()
            .reset_index(name='count'))
    total = ct.groupby('active_transport')['count'].transform('sum')
    ct['pct'] = ct['count']/total
    pivot = ct.pivot(index='active_transport', columns='unhealthy_cluster', values='pct').fillna(0)
    pivot.plot(kind='bar', stacked=True, figsize=(9,6), colormap='coolwarm')
    plt.legend(title='Unhealthy Cluster', labels=["No","Yes"], loc='upper right')
    plt.xlabel('Active Transport (0=Passive,1=Active)')
    plt.ylabel('Proportion')
    plt.title('Proportion of Unhealthy Cluster by Transport Type')
    save_fig("cluster_by_transport_stacked.png")

# plot interaction feature
if 'low_veg_high_screen' in df.columns:
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='target', hue='low_veg_high_screen', palette='magma')
    plt.xlabel('Obesity Class (Encoded)')
    plt.ylabel('Count')
    plt.legend(title='Low Veg & High Screen', labels=['No','Yes'])
    plt.title('Interaction: Low Vegetable Intake & High Screen Time by Class')
    save_fig("interaction_lowveg_highscreen_vs_class.png")

# plot correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
if 'target' in corr.columns:
    target_corr = (corr['target'].drop('target')
                   .reindex(corr.index.drop('target'))
                   .abs().sort_values(ascending=False))
    top_feats = target_corr.head(15).index.tolist() + ['target']
    cm = corr.loc[top_feats, top_feats]
    plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'shrink':0.6})
    plt.title('Correlation Matrix (Top 15 Features by |corr with target|)')
    save_fig("top_feature_correlations.png")

# plot distributions for behavioral features
selected_features = [c for c in ['tue','faf','fcvc','ch2o'] if c in df.columns]
if selected_features:
    n = len(selected_features)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, selected_features):
        sns.kdeplot(data=df, x=col, hue='target', common_norm=False, fill=True, alpha=0.4, ax=ax, legend=False)
        ax.set_title(fill(f"Distribution of {col} by class", 20))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title='Target', loc='upper right')
    fig.suptitle('Behavioral Feature Distributions', y=1.02, fontsize=16)
    fig.tight_layout()
    save_fig("behavioral_feature_distributions.png")

# pairplot for sampled data
subset_cols = [c for c in ['age','bmi','tue','faf','ch2o','target'] if c in df.columns]
if 'bmi' not in df.columns and {'weight','height'}.issubset(df.columns):
    df['bmi'] = df['weight'] / (df['height']**2)
    if 'bmi' not in subset_cols:
        subset_cols.insert(0,'bmi')
if len(subset_cols) >= 3:
    sample_df = df[subset_cols].sample(min(400, len(df)), random_state=42)
    g = sns.pairplot(sample_df, vars=[c for c in subset_cols if c != 'target'], hue='target', diag_kind='kde', corner=True, plot_kws={'alpha':0.5, 's':25, 'edgecolor':'none'})
    g.fig.suptitle('Pairwise Relationships (Sampled)', y=1.02)
    g.fig.savefig(os.path.join(FINAL_DIR, 'pairplot_sampled.png'), dpi=300, bbox_inches='tight')
    plt.close(g.fig)

# feature importance from random forest
rf_model_path = "output/models/random_forest_model.pkl"
if os.path.exists(rf_model_path):
    try:
        loaded = joblib.load(rf_model_path)
        if isinstance(loaded, dict):
            model_obj = loaded['model']
            feat_names = loaded.get('features') or []
        else:
            model_obj = loaded
            feat_names = [c for c in df.columns if c not in ['nobeyesdad','mtrans','target']]
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
            if len(importances) == len(feat_names):
                imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
                imp_df = imp_df.sort_values('importance', ascending=False).head(20)
                plt.figure(figsize=(10,7))
                sns.barplot(data=imp_df, x='importance', y='feature', palette='crest')
                plt.title('Random Forest Top 20 Feature Importances')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                save_fig('random_forest_feature_importance_top20.png')
    except Exception:
        pass

# done
print("###################### FINAL VISUALS SAVED TO output/final_visuals ######################")
