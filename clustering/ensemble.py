import os

import pandas as pd

from utils.constants import *

LIST_METHODS = [
    "outputs/kmeans_0.1/modernbert-embed-base",
    "outputs/kmeans_0.1/nomic-embed-text-v1.5",
]

DISTANCE_METRICS = [
    "euclidean",
    "euclidean",
]

OUTPUT_DIR = "outputs/kmeans_0.1/ensemble"

first_person_pronouns = [
    "i ",
    "me",
    "my",
    "mine",
    "myself",
    "i'm",
    "i've",
    "i'll",
    "i'd",
    "im ",
    "am ",
]


def filtering(text):
    for pronoun in first_person_pronouns:
        if pronoun in text.lower():
            return True
    return False


def avg_ensemble(symptom):
    # Read each prediction CSV file into a list of dataframes
    dfs = []
    for method, distance in zip(LIST_METHODS, DISTANCE_METRICS):
        # file_path = os.path.join(method, symptom, f"{distance}.csv")
        file_path = os.path.join(method, f"{symptom}.csv")
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        if "text" in df.columns and "doc_text" not in df.columns:
            df["doc_text"] = df["text"]
        df = df[df["doc_text"].apply(filtering)].reset_index(drop=True)
        df["similarity"] = df["similarity"] / df["similarity"].max()
        # print(method, symptom, len(df))
        print(df.head())
        dfs.append(df)

    # Combine all predictions into one dataframe
    combined = pd.concat(dfs, ignore_index=True)

    # Group by docid: average similarities, take the first text, and count occurrence per docid
    grouped = (
        combined.groupby("docid")
        .agg(similarity=("similarity", "mean"), text=("doc_text", "first"))
        .reset_index()
    )

    # Sort by descending average similarity and take top 1k
    # result = grouped.sort_values(by="similarity", ascending=False).head(1000)
    result = grouped.nlargest(5000, "similarity")

    return result


if __name__ == "__main__":
    for symptom in LIST_BDI_SYMPTOMS:
        # # AVG ensemble
        os.makedirs(f"{OUTPUT_DIR}_avg", exist_ok=True)
        result = avg_ensemble(symptom)
        result.to_csv(f"{OUTPUT_DIR}_avg/{symptom}.csv", index=False)
