import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utilities import (
    load_preprocessed_datasets,
    build_id_mapping,
    apply_id_mapping,
    add_user_reviews,
    add_item_reviews
)

from pathlib import Path
import sys
ROOT = Path.cwd().resolve().parent 
sys.path.append(str(ROOT))

# Configuration

CONFIG = {
    # BERT model
    "bert_model_name": "bert-base-uncased",

    # Review handling
    "max_user_reviews": 15,
    "max_item_reviews": 15,
    "max_tokens": 128,

    # Model dimensions
    "id_embed_dim": 64,
    "cnn_num_filters": 128,
    "cnn_kernel_sizes": [2, 3, 4],
    "text_projection_dim": 128,
    "mlp_hidden_dim": 256,

    # Training
    "batch_size": 32,
    "epochs": 7,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.25,

    "fine_tune_bert": False,

    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset and collate function
class RestaurantRatingDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "user_idx": int(row["user_idx"]),
            "item_idx": int(row["item_idx"]),
            "rating": float(row["rating"]),
            "user_reviews": row["user_reviews"],
            "item_reviews": row["item_reviews"],
        }


def collate_fn(batch):
    user_idx = torch.tensor([x["user_idx"] for x in batch], dtype=torch.long)
    item_idx = torch.tensor([x["item_idx"] for x in batch], dtype=torch.long)
    rating = torch.tensor([x["rating"] for x in batch], dtype=torch.float32)

    user_reviews = [x["user_reviews"] for x in batch]
    item_reviews = [x["item_reviews"] for x in batch]

    return {
        "user_idx": user_idx,
        "item_idx": item_idx,
        "rating": rating,
        "user_reviews": user_reviews,
        "item_reviews": item_reviews,
    }

# BERT + CNN text encoder
class BertCNNReviewEncoder(nn.Module):
    """
    Encodes a batch of examples.

    Each example contains multiple reviews.
    Each review is encoded independently with BERT + CNN.
    Then review vectors are aggregated into one profile vector per example.
    """

    def __init__(
        self,
        bert_model_name,
        cnn_num_filters,
        cnn_kernel_sizes,
        projection_dim,
        dropout,
        fine_tune_bert=True,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)

        self.bert_hidden_size = self.bert.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.bert_hidden_size,
                out_channels=cnn_num_filters,
                kernel_size=k
            )
            for k in cnn_kernel_sizes
        ])

        cnn_output_dim = cnn_num_filters * len(cnn_kernel_sizes)

        self.projection = nn.Sequential(
            nn.Linear(cnn_output_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_dim = projection_dim
        self.fine_tune_bert = fine_tune_bert

        if not fine_tune_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def encode_flat_reviews(self, flat_reviews, device, max_tokens):
        """
        flat_reviews: List[str]
        returns: Tensor [num_reviews, projection_dim]
        """

        if len(flat_reviews) == 0:
            return torch.empty(0, self.output_dim, device=device)

        encoded = self.tokenizer(
            flat_reviews,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if self.fine_tune_bert:
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            with torch.no_grad():
                bert_outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

        # token embeddings: [num_reviews, seq_len, hidden_size]
        token_embeddings = bert_outputs.last_hidden_state

        # Conv1d expects [batch, hidden_size, seq_len]
        x = token_embeddings.transpose(1, 2)

        conv_outputs = []

        for conv in self.convs:
            # [num_reviews, num_filters, seq_len - kernel_size + 1]
            c = torch.relu(conv(x))

            # Max pooling over token dimension
            # [num_reviews, num_filters]
            pooled = torch.max(c, dim=2).values

            conv_outputs.append(pooled)

        # [num_reviews, cnn_num_filters * num_kernels]
        review_vectors = torch.cat(conv_outputs, dim=1)

        # [num_reviews, projection_dim]
        review_vectors = self.projection(review_vectors)

        return review_vectors

    def forward(self, batch_reviews, device, max_tokens):
        """
        batch_reviews: List[List[str]]
        returns: Tensor [batch_size, projection_dim]
        """

        batch_size = len(batch_reviews)

        flat_reviews = []
        review_counts = []

        for reviews in batch_reviews:
            if not isinstance(reviews, list):
                reviews = []

            cleaned = [
                r for r in reviews
                if isinstance(r, str) and len(r.strip()) > 0
            ]

            review_counts.append(len(cleaned))
            flat_reviews.extend(cleaned)

        if len(flat_reviews) == 0:
            return torch.zeros(batch_size, self.output_dim, device=device)

        flat_vectors = self.encode_flat_reviews(
            flat_reviews=flat_reviews,
            device=device,
            max_tokens=max_tokens,
        )

        example_vectors = []
        start = 0

        for count in review_counts:
            if count == 0:
                example_vectors.append(
                    torch.zeros(self.output_dim, device=device)
                )
            else:
                vectors = flat_vectors[start:start + count]

                # Mean aggregation over review vectors.
                # You can replace this with attention later.
                profile_vector = vectors.mean(dim=0)

                example_vectors.append(profile_vector)
                start += count

        return torch.stack(example_vectors, dim=0)



# Full hybrid recommender model

class HybridBertCNNRecommender(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        id_embed_dim,
        bert_model_name,
        cnn_num_filters,
        cnn_kernel_sizes,
        text_projection_dim,
        mlp_hidden_dim,
        dropout,
        fine_tune_bert=True,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, id_embed_dim)
        self.item_embedding = nn.Embedding(num_items, id_embed_dim)

        self.user_text_encoder = BertCNNReviewEncoder(
            bert_model_name=bert_model_name,
            cnn_num_filters=cnn_num_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            projection_dim=text_projection_dim,
            dropout=dropout,
            fine_tune_bert=fine_tune_bert,
        )

        self.item_text_encoder = BertCNNReviewEncoder(
            bert_model_name=bert_model_name,
            cnn_num_filters=cnn_num_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            projection_dim=text_projection_dim,
            dropout=dropout,
            fine_tune_bert=fine_tune_bert,
        )

        input_dim = (
            id_embed_dim
            + id_embed_dim
            + text_projection_dim
            + text_projection_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_hidden_dim // 2, 1)
        )

    def forward(self, user_idx, item_idx, user_reviews, item_reviews, device, max_tokens):
        user_id_vec = self.user_embedding(user_idx)
        item_id_vec = self.item_embedding(item_idx)

        user_text_vec = self.user_text_encoder(
            batch_reviews=user_reviews,
            device=device,
            max_tokens=max_tokens,
        )

        item_text_vec = self.item_text_encoder(
            batch_reviews=item_reviews,
            device=device,
            max_tokens=max_tokens,
        )

        x = torch.cat([user_id_vec, item_id_vec, user_text_vec, item_text_vec], dim=1)

        rating_pred = self.mlp(x).squeeze(1)

        return rating_pred

# Evaluation function
def evaluate(model, dataloader, device, max_tokens):
    model.eval()

    preds = []
    true_ratings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            user_idx = batch["user_idx"].to(device)
            item_idx = batch["item_idx"].to(device)
            rating = batch["rating"].to(device)

            preds = model(
                user_idx=user_idx,
                item_idx=item_idx,
                user_reviews=batch["user_reviews"],
                item_reviews=batch["item_reviews"],
                device=device,
                max_tokens=max_tokens,
            )

            # Optional clamp to valid rating range.
            preds = torch.clamp(preds, min=1.0, max=5.0)

            preds.extend(preds.cpu().numpy())
            true_ratings.extend(rating.cpu().numpy())

    rmse = root_mean_squared_error(true_ratings, preds)
    mae = mean_absolute_error(true_ratings, preds)

    return rmse, mae


if __name__ == "__main__":
    set_seed(CONFIG["seed"])

    train_df, val_df, test_df = load_preprocessed_datasets()

    UNK_USER = "<UNK_USER>"
    UNK_ITEM = "<UNK_ITEM>"
    user2idx = build_id_mapping(train_df["user_id"], UNK_USER)
    item2idx = build_id_mapping(train_df["business_id"], UNK_USER)

    num_users = len(user2idx)
    num_items = len(item2idx)

    train_df = apply_id_mapping(train_df, user2idx, item2idx)
    val_df = apply_id_mapping(val_df, user2idx, item2idx)
    test_df = apply_id_mapping(test_df, user2idx, item2idx)

    train_df = add_user_reviews(train_df, CONFIG["max_user_reviews"])
    val_df = add_user_reviews(val_df, CONFIG["max_user_reviews"])
    test_df = add_user_reviews(test_df, CONFIG["max_user_reviews"])

    val_df["row_id"] = None
    test_df["row_id"] = None

    train_df = add_item_reviews(train_df, CONFIG["max_item_reviews"])
    val_df = add_item_reviews(val_df, CONFIG["max_item_reviews"])
    test_df = add_item_reviews(test_df, CONFIG["max_item_reviews"])

    train_dataset = RestaurantRatingDataset(train_df)
    val_dataset = RestaurantRatingDataset(val_df)
    test_dataset = RestaurantRatingDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Training loop
    device = CONFIG["device"]

    model = HybridBertCNNRecommender(
        num_users=num_users,
        num_items=num_items,
        id_embed_dim=CONFIG["id_embed_dim"],
        bert_model_name=CONFIG["bert_model_name"],
        cnn_num_filters=CONFIG["cnn_num_filters"],
        cnn_kernel_sizes=CONFIG["cnn_kernel_sizes"],
        text_projection_dim=CONFIG["text_projection_dim"],
        mlp_hidden_dim=CONFIG["mlp_hidden_dim"],
        dropout=CONFIG["dropout"],
        fine_tune_bert=CONFIG["fine_tune_bert"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_model_path = ROOT/"trained_models/hybrid_bert_cnn.pt"

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            user_idx = batch["user_idx"].to(device)
            item_idx = batch["item_idx"].to(device)
            rating = batch["rating"].to(device)

            optimizer.zero_grad()

            preds = model(
                user_idx=user_idx,
                item_idx=item_idx,
                user_reviews=batch["user_reviews"],
                item_reviews=batch["item_reviews"],
                device=device,
                max_tokens=CONFIG["max_tokens"],
            )

            loss = criterion(preds, rating)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)

        val_rmse, val_mae = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            max_tokens=CONFIG["max_tokens"],
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model.")


    # Test evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_rmse, test_mae = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        max_tokens=CONFIG["max_tokens"],
    )

    result = {
        "rmse": test_rmse,
        "mae": test_mae
    }
    
    print("-" * 50)
    print("Final Test Results")
    print("Test RMSE:", test_rmse)
    print("Test MAE:", test_mae)
    print("-" * 50)

    result_df = pd.DataFrame([result])
    result_df.to_csv(ROOT/"results/hybrid_bert_cnn.csv", index=False)

