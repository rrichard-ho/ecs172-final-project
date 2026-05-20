import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load data
BASE_PATH = Path(__file__).resolve().parent / "data"

train_df = pd.read_csv(BASE_PATH / "train.csv")
test_df  = pd.read_csv(BASE_PATH / "test.csv")

train_df = train_df[["user_id", "business_id", "rating"]].copy()
test_df  = test_df[["user_id", "business_id", "rating"]].copy()

# index mapping
user_ids = train_df["user_id"].unique()
business_ids = train_df["business_id"].unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
business_to_idx = {b: i for i, b in enumerate(business_ids)}

train_df["user_idx"] = train_df["user_id"].map(user_to_idx)
train_df["business_idx"] = train_df["business_id"].map(business_to_idx)

test_df["user_idx"] = test_df["user_id"].map(user_to_idx)
test_df["business_idx"] = test_df["business_id"].map(business_to_idx)

# remove unknown users/items in test
test_df = test_df.dropna(subset=["user_idx", "business_idx"]).copy()
test_df["user_idx"] = test_df["user_idx"].astype(int)
test_df["business_idx"] = test_df["business_idx"].astype(int)

# hyper-parameters
n_users = len(user_ids)
n_items = len(business_ids)
n_factors = 20
lr = 0.01
reg = 0.02
epochs = 20

# initialize latent factors
np.random.seed(42)
U = np.random.normal(scale=0.1, size=(n_users, n_factors))
V = np.random.normal(scale=0.1, size=(n_items, n_factors))

# train using SGD
for epoch in range(epochs):
    shuffled = train_df.sample(frac=1, random_state=epoch)

    for row in shuffled.itertuples():
        u = row.user_idx
        i = row.business_idx
        r = row.rating

        pred = np.dot(U[u], V[i])
        err = r - pred

        u_vec = U[u].copy()
        U[u] += lr * (err * V[i] - reg * U[u])
        V[i] += lr * (err * u_vec - reg * V[i])

    print(f"Epoch {epoch + 1}/{epochs} complete")

# evaluation
preds = []
actuals = []

for row in test_df.itertuples():
    p = np.dot(U[row.user_idx], V[row.business_idx])
    preds.append(np.clip(p, 1, 5))
    actuals.append(row.rating)

rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = mean_absolute_error(actuals, preds)

print(f"\nFinal RMSE: {rmse:.4f}")
print(f"Final MAE : {mae:.4f}")