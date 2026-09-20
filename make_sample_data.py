"""Generate a synthetic loan-approval dataset with deliberately planted problems.

Planted issues (so we know DataSentry should find them):
  * gender bias: women are approved less often at the same income/credit level
  * race is not a model input, but `zip_code_group` is a PROXY for race
  * ~8% missing values in income and employment_years
  * ~2% exact duplicate rows
  * a few absurd outliers in income and age
  * a constant column (country) and an ID column
  * target imbalance: approvals are the minority class

Run:  python make_sample_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

gender = rng.choice(["Male", "Female"], size=N, p=[0.68, 0.32])  # under-representation of women
race = rng.choice(["Group_A", "Group_B", "Group_C"], size=N, p=[0.70, 0.22, 0.08])
age = np.clip(rng.normal(38, 11, N), 18, 75).round().astype(int)

# zip_code_group is strongly tied to race -> proxy variable
zip_map = {"Group_A": ["Z1", "Z2"], "Group_B": ["Z3", "Z4"], "Group_C": ["Z5", "Z4"]}
zip_code_group = [rng.choice(zip_map[r], p=[0.85, 0.15]) for r in race]

education = rng.choice(["HighSchool", "Bachelors", "Masters", "PhD"], size=N, p=[0.40, 0.38, 0.17, 0.05])
edu_boost = pd.Series(education).map({"HighSchool": 0, "Bachelors": 8000, "Masters": 16000, "PhD": 24000}).values
income = np.clip(rng.normal(42000, 12000, N) + edu_boost + (gender == "Male") * 3000, 12000, None).round()
credit_score = np.clip(rng.normal(640, 70, N), 300, 850).round().astype(int)
employment_years = np.clip(rng.normal(8, 5, N), 0, 40).round(1)
loan_amount = np.clip(rng.normal(18000, 8000, N), 1000, None).round()

# True "merit" signal + planted discrimination
score = (
    (credit_score - 640) / 70 * 0.9
    + (income - 42000) / 12000 * 0.7
    + employment_years / 8 * 0.2
    - loan_amount / 18000 * 0.4
    - 0.9 * (gender == "Female")            # planted gender bias
    - 0.5 * (race == "Group_B")             # planted race bias (via proxy zip)
    - 0.5 * (race == "Group_C")
    - 1.4                                   # makes approvals the minority class
)
prob = 1 / (1 + np.exp(-score))
approved = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "applicant_id": np.arange(1, N + 1),
    "gender": gender,
    "race": race,
    "age": age,
    "education": education,
    "zip_code_group": zip_code_group,
    "income": income,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "loan_amount": loan_amount,
    "country": "USA",
    "loan_approved": approved,
})

# Data quality problems
for col, frac in [("income", 0.08), ("employment_years", 0.08), ("education", 0.03)]:
    idx = rng.choice(N, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan
df.loc[rng.choice(N, 6, replace=False), "income"] = 9_999_999      # outliers
df.loc[rng.choice(N, 4, replace=False), "age"] = 190               # impossible ages
dupes = df.sample(int(N * 0.02), random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df.to_csv("sample_data/loan_approvals_synthetic.csv", index=False)
print("Saved sample_data/loan_approvals_synthetic.csv", df.shape)
print(df.groupby("gender")["loan_approved"].mean().round(3))
