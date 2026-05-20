from collections import defaultdict
import pandas as pd
from pathlib import Path
import sys

ROOT = Path.cwd().resolve().parent 
sys.path.append(str(ROOT))

# Load data
def load_preprocessed_datasets():

    train_df = pd.read_csv(ROOT/"data/train.csv")
    val_df = pd.read_csv(ROOT/"data/val.csv")
    test_df = pd.read_csv(ROOT/"data/test.csv")

    cols = ["user_id", "business_id", "rating", "review_text", "history_reviews"]

    train_df = train_df[cols].copy()
    val_df = val_df[cols].copy()
    test_df = test_df[cols].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df

# Build ID - index mappings
def build_id_mapping(values, unk_token):
    unique_values = sorted(values.unique().tolist())
    mapping = {unk_token: 0}

    for idx, value in enumerate(unique_values, start=1):
        mapping[value] = idx

    return mapping

def apply_id_mapping(df, user_to_idx, item_to_idx):
    df = df.copy()
    df["user_idx"] = df["user_id"].map(lambda x: user_to_idx.get(x, 0))
    df["item_idx"] = df["business_id"].map(lambda x: item_to_idx.get(x, 0))
    return df

# Build user review lists
# For each interaction, user text comes from history_reviews.
# We don't use the target review_text as user input.

def extract_history_reviews(history_reviews, max_reviews):
    if not isinstance(history_reviews, list):
        return []

    reviews = []

    for entry in history_reviews[:max_reviews]:
        if isinstance(entry, list) and len(entry) >= 2:
            text = entry[1]
            if isinstance(text, str) and len(text.strip()) > 0:
                reviews.append(text.strip())

    return reviews


def add_user_reviews(df, max_user_reviews):
    df = df.copy()
    df["user_reviews"] = df["history_reviews"].apply(
        lambda x: extract_history_reviews(x, max_user_reviews)
    )
    return df


def get_item_reviews_for_row( item_to_train_reviews, business_id, split, row_id=None, max_reviews=10):
    reviews = item_to_train_reviews.get(business_id, [])

    selected = []

    for r in reviews:
        # During training, exclude the current row's own review.
        if split == "train" and row_id is not None and r["row_id"] == row_id:
            continue

        selected.append(r["text"])

        if len(selected) >= max_reviews:
            break

    return selected


def add_item_reviews(df, max_item_reviews):
    df = df.copy()

    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    item_to_train_reviews = defaultdict(list)

    for _, row in df.iterrows():
        business_id = row["business_id"]
        review_text = row["review_text"]

        if isinstance(review_text, str) and len(review_text.strip()) > 0:
            item_to_train_reviews[business_id].append(
                {
                    "row_id": row["row_id"],
                    "text": review_text.strip()
                }
            )

    item_reviews = []

    for _, row in df.iterrows():
        row_id = row["row_id"] if "row_id" in row else None

        reviews = get_item_reviews_for_row(
            item_to_train_reviews=item_to_train_reviews,
            business_id=row["business_id"],
            split=row["split"],
            row_id=row_id,
            max_reviews=max_item_reviews,
        )

        item_reviews.append(reviews)

    df["item_reviews"] = item_reviews

    return df