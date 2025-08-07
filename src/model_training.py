import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

df = pd.read_csv("data/obesity_features.csv")

target = 'target'
exclude = ['nobeyesdad', 'mtrans']
features = [col for col in df.columns if col not in exclude and col != target]

X = df[features]
if any(X.dtypes == 'object'):
    X = pd.get_dummies(X, drop_first=True)
feature_cols = X.columns.tolist()
y = df[target]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

os.makedirs("output/models", exist_ok=True)

# random forest
print("###################### RANDOM FOREST ######################")
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(classification_report(y_test, rf_preds))
print("confusion matrix:")
print(confusion_matrix(y_test, rf_preds))
joblib.dump({'model': rf, 'features': feature_cols}, "output/models/random_forest_model.pkl")

print()

# xgboost
print("###################### XGBOOST ######################")
xgb = XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print(classification_report(y_test, xgb_preds))
print("confusion matrix:")
print(confusion_matrix(y_test, xgb_preds))
joblib.dump({'model': xgb, 'features': feature_cols}, "output/models/xgboost_model.pkl")
joblib.dump({'model': xgb, 'features': feature_cols}, "output/models/xgboost_model.pkl")
