from utilities import load_preprocessed_datasets
import pandas as pd

from libreco.data import DatasetPure
from libreco.algorithms import NCF
from libreco.evaluation import evaluate



from pathlib import Path
import sys
ROOT = Path.cwd().resolve().parent 
sys.path.append(str(ROOT))

def transform_df(df: pd.DataFrame):
    df = df.copy()
    new_df = df[["user_id", "business_id", "rating"]].rename(
        columns={
            "user_id": "user",
            "business_id": "item",
            "rating": "label"
        }
    )
    return new_df

if __name__ == "__main__":
    train_df, val_df, test_df = load_preprocessed_datasets()
    train_cf = transform_df(train_df)
    val_cf = transform_df(val_df)

    train_data, data_info = DatasetPure.build_trainset(train_cf)
    eval_data = DatasetPure.build_evalset(val_cf)

    ncf = NCF(
        task="rating",
        data_info=data_info,
        loss_type="mse",
        embed_size=64,
        n_epochs=10,
        lr=1e-3,
        batch_size=32,
    )

    ncf.fit(
        train_data,
        eval_data=eval_data,
        neg_sampling=False,
        metrics=["mse"]
    )

    data_info.save(path=ROOT/"trained_models", model_name="ncf_model")
    ncf.save(path=ROOT/"trained_models", model_name="ncf_model")

    test_cf = transform_df(test_df)
    test_data = DatasetPure.build_testset(test_cf)
    metrics = evaluate(
        model=ncf,
        data=test_cf,
        neg_sampling=False,
        metrics=["rmse", "mae"]
    )

    print(metrics)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(ROOT/"results/ncf_metrics.csv", index=False)

    preds = ncf.predict(test_cf["user"], test_cf["item"])
    preds_ratings = test_df[["user_id", "business_id"]].copy()
    preds_ratings["predicted_rating"] = preds
    preds_ratings.to_csv(ROOT/"results/ncf_prediction.csv", index=False)