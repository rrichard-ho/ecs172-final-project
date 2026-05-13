import json
import pandas as pd
from pathlib import Path

# loading data
DATA_PATH = Path(__file__).resolve().parent / "filter_all_t.json"

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    train_data = json.load(f)["train"]

# creating dataframe
rows = []
for entry in train_data:
    base = {
        "business_id": entry.get("business_id"),
        "user_id":     entry.get("user_id"),
        "rating":      entry.get("rating"),
        "review_text": entry.get("review_text"),
        "pics":        entry.get("pics"),
    }
    history = entry.get("history_reviews") or []
    if history:
        for hist_id, hist_text in history:
            rows.append({**base, "history_review_id": hist_id, "history_review_text": hist_text})
    else:
        rows.append(base)

df = pd.DataFrame(rows)

# print first three for readability
for _, row in df.head(100).iterrows():
    print(f"""
business_id  : {row['business_id']}
user_id      : {row['user_id']}
rating       : {row['rating']}
review       : {row['review_text']}
history id   : {row.get('history_review_id', 'N/A')}
history text : {row.get('history_review_text', 'N/A')}
{"─" * 80}""")