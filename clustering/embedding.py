import os

import pandas as pd
from sentence_transformers import SentenceTransformer


def read_data(data_path, text_column="doc_text", id_column="docid"):
    print(f"Reading data from {data_path}")

    if os.path.isdir(data_path):
        dfs = []
        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(data_path, file))
                    df = df.dropna(subset=[text_column])
                    df[text_column] = df[text_column].astype(str)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading file {file}: {e}")

        concat_dfs = pd.concat(dfs, ignore_index=True)

    elif data_path.endswith(".csv"):
        concat_dfs = pd.read_csv(data_path)
        concat_dfs = concat_dfs.dropna(subset=[text_column])
        concat_dfs[text_column] = concat_dfs[text_column].astype(str)

    # drop text with less than 5 characters
    concat_dfs = concat_dfs[concat_dfs[text_column].str.len() >= 5]
    # drop text that only contains numbers
    concat_dfs = concat_dfs[~concat_dfs[text_column].str.isnumeric()]
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
