import os

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from extractor.model import get_model

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def preprocess(df: pd.DataFrame):
    df = df.copy()
    df = df.set_index("docid")
    df = df.replace(-1, 0)

    # label: array of all symptom columns (except text)
    df["label"] = df.drop(["text"], axis=1).values.tolist()
    df["label"] = df["label"].apply(lambda x: [float(val) for val in x])

    return df[["text", "label"]]


def prepare_dataset(data_dir):
    data_23 = pd.read_csv(os.path.join(data_dir, "erisk2023_majority.csv"))
    data_24 = pd.read_csv(os.path.join(data_dir, "erisk2024_majority.csv"))

    val_df = data_24.sample(int(len(data_24) * 0.2), random_state=SEED)
    train_df = pd.concat(
        [data_23, data_24[~data_24["doc_id"].isin(val_df["doc_id"])]]
    ).reset_index(drop=True)

    train_df = preprocess(train_df)
    val_df = preprocess(val_df)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
        }
    )

    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to multi-label predictions
    pred_probas = torch.sigmoid(torch.tensor(logits))
    predictions = (pred_probas > 0.5).int().numpy()

    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    return {
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


def train_model(model, dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    return trainer


def main():
    data_dir = "data/erisk25-t1-processed"
    model_path = "rafalposwiata/deproberta-large-depression"

    dataset = prepare_dataset(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=256
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = get_model(model_path=model_path, num_labels=21)

    trainer = train_model(model, tokenized_datasets)
    trainer.train()

    torch.cuda.empty_cache()
    trainer.evaluate()


if __name__ == "__main__":
    main()
