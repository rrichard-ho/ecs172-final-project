import json
import pandas as pd
from collections import defaultdict
import numpy as np
from pathlib import Path
import sys

ROOT = Path.cwd().resolve().parent 
sys.path.append(str(ROOT))
DATA_PATH = ROOT/"data/filter_all_t.json"

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    train_df = pd.DataFrame(train_data).drop(columns=['pics'])
    val_df = pd.DataFrame(val_data).drop(columns='pics')
    test_df = pd.DataFrame(test_data).drop(columns='pics')

    data = pd.concat([train_df, val_df, test_df])

    return data

def val_test_split(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)

    protected = set()

    # Keep at least one interaction per user
    for _, group in df.groupby("user_id"):
        protected.add(rng.choice(group.index.to_numpy()))

    # Keep at least one interaction per item
    for _, group in df.groupby("business_id"):
        protected.add(rng.choice(group.index.to_numpy()))

    protected = np.array(sorted(protected))
    remaining = np.array([i for i in df.index if i not in protected])

    rng.shuffle(remaining)

    n_total = len(df)
    n_val = int(round(val_ratio * n_total))
    n_test = int(round(test_ratio * n_total))

    test_idx = remaining[:n_test]
    val_idx = remaining[n_test:n_test + n_val]
    extra_train_idx = remaining[n_test + n_val:]

    train_idx = np.concatenate([protected, extra_train_idx])

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df

def utility_mappings(df: pd.DataFrame):
    """
    create two dictionaries that
        + map user_id to user index
        + map business_id to item index
    """
    df = df.copy()
    user_id_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(df['user_id'].unique())
    }

    item_id_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(df['business_id'].unique())
    }

    return user_id_to_idx, item_id_to_idx

def map_id_to_idx(df: pd.DataFrame, user_id_to_idx, item_id_to_idx):
    """
    create extra columns to to denote user index and item index for each pair <user, item>
    """
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_id_to_idx)
    df["business_idx"] = df["business_id"].map(item_id_to_idx)
    return df

def build_user_history_reviews(df: pd.DataFrame, max_reviews=20):
    user_reviews = defaultdict(list)
    for _, row in df.iterrows():
        user_reviews[row["user_idx"]].append(row["review_text"])
    
    user_profile_text = {
        user_idx: " ".join(reviews[:max_reviews])
        for user_idx, reviews in user_reviews.items()
    }

    return user_profile_text

def build_restaurant_history_reviews(df: pd.DataFrame, max_reviews=20):
    """
    for each restaurant, aggregate all reviews about it into one single text
    """
    rest_reviews = defaultdict(list)
    for _, row in df.iterrows():
        rest_reviews[row["business_idx"]].append(row["review_text"])
    
    restaurant_profile_text = {
        restaurant_idx: " ".join(reviews[:max_reviews])
        for restaurant_idx, reviews in rest_reviews.items()
    }

    return restaurant_profile_text

ROOT = Path.cwd().resolve().parent 
sys.path.append(str(ROOT))
DATA_PATH = ROOT/"data/filter_all_t.json"

if __name__ == "__main__":
    data = load_data(path=DATA_PATH)
    train_df, val_df, test_df = val_test_split(data)
    print(len(train_df))
    print(len(val_df))
    print(len(test_df))
    train_df.to_csv(ROOT/"data/train.csv")
    val_df.to_csv(ROOT/"data/val.csv")
    test_df.to_csv(ROOT/"data/test.csv")