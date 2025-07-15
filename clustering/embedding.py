import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def read_data(data_path):
    print(f"Reading data from {data_path}")

    if os.path.isdir(data_path):
        dfs = []
        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(data_path, file))
                    df = df.dropna(subset=["doc_text"])
                    df["doc_text"] = df["doc_text"].astype(str)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

        concat_dfs = pd.concat(dfs, ignore_index=True)
        if "doc_no" in concat_dfs.columns and "docid" not in concat_dfs.columns:
            concat_dfs["docid"] = concat_dfs["doc_no"]
            concat_dfs.drop("doc_no", axis=1, inplace=True)

    elif data_path.endswith(".csv"):
        concat_dfs = pd.read_csv(data_path)
        if "text" in concat_dfs.columns and "doc_text" not in concat_dfs.columns:
            concat_dfs.rename(columns={"text": "doc_text"}, inplace=True)
        concat_dfs = concat_dfs.dropna(subset=["doc_text"])
        concat_dfs["doc_text"] = concat_dfs["doc_text"].astype(str)

    # drop text with less than 5 characters
    concat_dfs = concat_dfs[concat_dfs["doc_text"].str.len() >= 5]
    # drop text that only contains numbers
    concat_dfs = concat_dfs[~concat_dfs["doc_text"].str.isnumeric()]
    return concat_dfs.reset_index(drop=True)


def encoding(model: SentenceTransformer, device, df, batch_size=64):
    """Encode text data using SentenceTransformer model."""
    print(f"Using device: {device}")

    # Load data and model
    sentences = df["doc_text"].values

    embeddings = model.encode(
        sentences, show_progress_bar=True, batch_size=batch_size, device=device
    )
    print(embeddings.shape)

    return embeddings
