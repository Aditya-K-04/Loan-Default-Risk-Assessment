from ucimlrepo import fetch_ucirepo
import pandas as pd

credit_approval = fetch_ucirepo(id=27)

X = credit_approval.data.features
y = credit_approval.data.targets

df = X.copy()
target_col = y.columns[0]

# Convert target to binary
# + -> 0, - -> 1
df["TARGET"] = y[target_col].apply(lambda x: 1 if str(x).strip() == "-" else 0)

df.to_csv("loan_data.csv", index=False)

print("loan_data.csv created successfully")
print(df.head())
print(df["TARGET"].value_counts())