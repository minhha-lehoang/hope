import torch
import pandas as pd
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_similarity,
)
from sklearn.cluster import KMeans


def minkowski_distance(a, b):
    return torch.nn.functional.pairwise_distance(a, b, p=float("inf"))


def similarity(a, b, metric="cosine"):
    if metric == "cosine":
        return cosine_similarity(a, b)
    elif metric == "euclidean":
        # return euclidean_distance(a, b)
        return euclidean_distances(a, b)
    elif metric == "manhattan":
        # return manhattan_distance(a, b)
        return manhattan_distances(a, b)
    elif metric == "minkowski":
        return minkowski_distance(a, b)
    else:
        raise ValueError(f"Invalid metric: {metric}")


def k_means_clustering(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_


def calculate_distances(sentence_embedding, query_embeddings, metric="minkowski"):
    # return distance between sentence embedding and nearest query embedding
    return (
        similarity(
            torch.Tensor([sentence_embedding]),
            torch.Tensor(query_embeddings),
            metric=metric,
        )
        .min()  # get the minimum distance to any query embedding
        .numpy()
    )


def get_top_sentences_for_symptoms(
    df: pd.DataFrame, symptom, query_embeddings, top_k, distance_metric="euclidean"
):
    symptom_df = df[df["symptom"] == symptom].copy()
    symptom_df["distance"] = df["embedding"].apply(
        lambda x: calculate_distances(x, query_embeddings, distance_metric)
    )
    print(symptom_df.head())
    symptom_df["similarity"] = 1 - symptom_df["distance"] / symptom_df["distance"].max()
    # symptom_df = symptom_df.sort_values("similarity", ascending=False)
    # symptom_df = symptom_df.head(top_k)
    symptom_df = symptom_df.nlargest(top_k, "similarity")
    return symptom_df[["docid", "doc_text", "similarity"]]
