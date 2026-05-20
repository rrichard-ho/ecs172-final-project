import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load data
DATA_PATH = Path(__file__).resolve().parent / "filter_all_t.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)["train"]

rows = [
    {
        "business_id": entry.get("business_id"),
        "user_id":     entry.get("user_id"),
        "rating":      float(entry.get("rating")),
    }
    for entry in train_data
]
df = pd.DataFrame(rows)

# index mapping
user_ids      = df["user_id"].unique()
business_ids  = df["business_id"].unique()
user_to_idx   = {u: i for i, u in enumerate(user_ids)}
business_to_idx = {b: i for i, b in enumerate(business_ids)}

df["user_idx"]     = df["user_id"].map(user_to_idx)
df["business_idx"] = df["business_id"].map(business_to_idx)

# split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_users = set(train_df["user_idx"])
train_items = set(train_df["business_idx"])
test_df = test_df[
    test_df["user_idx"].isin(train_users) &
    test_df["business_idx"].isin(train_items)
].copy()

# hyper-parameters
n_users = len(user_ids)
n_items = len(business_ids)
n_factors = 20
lr = 0.01
reg = 0.02
epochs = 20

# initialize latent terms
np.random.seed(42)
U   = np.random.normal(scale=0.1, size=(n_users, n_factors))
V   = np.random.normal(scale=0.1, size=(n_items, n_factors))

# train using SGD
for epoch in range(epochs):
    shuffled = train_df.sample(frac=1, random_state=epoch)

    for row in shuffled.itertuples():
        u = row.user_idx
        i = row.business_idx
        r = row.rating

        pred = np.dot(U[u], V[i])
        err  = r - pred

        # update
        u_vec = U[u].copy()
        U[u] += lr * (err * V[i]  - reg * U[u])
        V[i] += lr * (err * u_vec - reg * V[i])

  
    print(f"Epoch {epoch + 1}/{epochs} complete")

# calculate RMSE
preds   = []
actuals = []
for row in test_df.itertuples():
    p = np.dot(U[row.user_idx], V[row.business_idx])
    preds.append(np.clip(p, 1, 5))
    actuals.append(row.rating)

rmse = np.sqrt(mean_squared_error(actuals, preds))
mae  = mean_absolute_error(actuals, preds)
print(f"\nFinal RMSE: {rmse:.4f}")
print(f"Final MAE : {mae:.4f}")