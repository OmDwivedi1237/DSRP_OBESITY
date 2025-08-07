import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/obesity_cleaned.csv")

df['low_veg'] = (df['fcvc'] < 2).astype(int)
df['high_screen'] = (df['tue'] > 2).astype(int)
df['low_veg_high_screen'] = (df['low_veg'] & df['high_screen']).astype(int)

df['high_calorie_food'] = df['favc'].map({'yes': 1, 'no': 0}).astype(int)
df['snacks_often'] = df['caec'].isin(['Frequently', 'Always']).astype(int)
df['low_water'] = (df['ch2o'] < 1.5).astype(int)
df['low_exercise'] = (df['faf'] < 1.5).astype(int)

df['unhealthy_cluster'] = ((df['low_veg'] + df['high_screen'] + df['high_calorie_food'] + df['snacks_often'] + df['low_water'] + df['low_exercise']) >= 4).astype(int)

df['active_transport'] = df['transport_type'].map({'Active': 1, 'Passive': 0}).astype(int)

le = LabelEncoder()
df['target'] = le.fit_transform(df['nobeyesdad'])

df = pd.get_dummies(df, columns=['gender', 'family_history_with_overweight', 'smoke', 'scc', 'calc', 'transport_type'], drop_first=True)

df.to_csv("data/obesity_features.csv", index=False)
print("###################### FEATURES READY ######################")