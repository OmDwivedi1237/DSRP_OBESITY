import shap
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

df = pd.read_csv("data/obesity_features.csv")

dumped = joblib.load("output/models/xgboost_model.pkl")
if isinstance(dumped, dict) and 'model' in dumped:
    model = dumped['model']
    model_features = dumped['features']
else:
    model = dumped
    model_features = [c for c in df.columns if c not in ['nobeyesdad', 'mtrans', 'target']]

raw_X = df.drop(columns=['nobeyesdad', 'mtrans', 'target'])
for col in model_features:
    if col not in raw_X.columns:
        raw_X[col] = 0
X = raw_X[model_features].copy()

dtypes_object = X.select_dtypes(include=['object']).columns
if len(dtypes_object) > 0:
    X = pd.get_dummies(X, drop_first=True)
    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    X = X[model_features]

X = X.fillna(0)

explainer = shap.TreeExplainer(model)
try:
    shap_values = explainer.shap_values(X, check_additivity=False)
except TypeError:
    shap_values = explainer.shap_values(X)

# make output folder
os.makedirs("output/plots", exist_ok=True)

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("output/plots/shap_summary_bar.png")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("output/plots/shap_summary_beeswarm.png")
plt.close()

# done
print("###################### SHAP PLOTS SAVED ######################")
