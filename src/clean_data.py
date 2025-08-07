import pandas as pd

df = pd.read_csv('data/obesity.csv')

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

categorical_cols = ['gender', 'family_history_with_overweight', 'favc', 'caec', 'smoke', 'scc', 'calc', 'mtrans', 'nobeyesdad']
for col in categorical_cols:
    df[col] = df[col].astype('category')

def simplify_transport(x):
    if x in ['Walking', 'Bike']:
        return 'Active'
    else:
        return 'Passive'

df['transport_type'] = df['mtrans'].apply(simplify_transport)
df['transport_type'] = df['transport_type'].astype('category')

df.to_csv("data/obesity_cleaned.csv", index=False)
print("###################### CLEANED DAT ######################")
print("###################### CLEANED DAT ######################")
