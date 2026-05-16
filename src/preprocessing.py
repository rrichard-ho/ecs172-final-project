import json
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

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
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Choose one row per user that must go to train
    mandatory_train = (
        df.groupby("user_id", group_keys=False)
          .sample(n=1, random_state=seed)
    )

    # Remove mandatory part 
    mandatory_idx = set(mandatory_train.index)
    remainder = df.drop(index=mandatory_idx)

    n_total = len(df)
    target_train_size = int(round(train_ratio * n_total))
    target_val_size = int(round(val_ratio * n_total))
    target_test_size = n_total - target_train_size - target_val_size

    # Add enough remainder rows to train to approach @train_ratio
    additional_train_needed = max(0, target_train_size - len(mandatory_train))
    remainder = remainder.sample(frac=1, random_state=seed)
    additional_train = remainder.iloc[:additional_train_needed]
    # Complete train set
    train_df = pd.concat([mandatory_train, additional_train])
    
    leftover = remainder.iloc[additional_train_needed:]
    # Split leftover globally into validation and test
    val_df = leftover.iloc[:target_val_size]
    test_df = leftover.iloc[target_val_size:target_val_size + target_test_size]

    # Shuffle final splits
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

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
    """
    for each user, aggregate all reviews written by that user into one single text
    """
    # def helper(history_reviews, max_revies=15):
    #     if not isinstance(history_reviews, list):
    #         return ""
    #     texts = []
    #     for item in history_reviews[:max_revies]:
    #         if isinstance(item, list) and len(item) >= 2:
    #             texts.append(item[1])
    #     return "".join(texts)
    
    # user_profile_text = {}
    # for _, row in df.iterrows():
    #     user_profile_text[row["user_idx"]] = helper(row["history_reviews"])

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