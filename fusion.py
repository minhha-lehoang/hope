import pandas as pd
import os
from glob import glob
import argparse


def normalize(df, column):
    """
    Normalize a numeric column by dividing by its max value.
    Handles columns where the max is 0 to avoid division by zero.
    """
    max_val = df[column].max()
    if max_val > 0:
        df[column] = df[column] / max_val
    return df


def fuse_scores(
    supervised_df, unsupervised_df, alpha=0.5, beta=0.5, symptom_id=None
):
    """
    Fuses supervised and unsupervised scores for a single symptom.
    """
    # Merge scores
    merged_df = pd.merge(
        supervised_df,
        unsupervised_df,
        on="doc_id",
        how="outer",
        suffixes=("_sup", "_unsup"),
    ).fillna(0)

    score_sup = f"query_{symptom_id}_sup"
    score_unsup = f"query_{symptom_id}_unsup"

    # Additive Fusion
    merged_df["score_add"] = (alpha * merged_df[score_sup]) + (
        (1 - alpha) * merged_df[score_unsup]
    )

    # Maximum Fusion
    merged_df["score_max"] = merged_df[[score_sup, score_unsup]].max(axis=1)

    # Final Score
    merged_df["final_score"] = (beta * merged_df["score_add"]) + (
        (1 - beta) * merged_df["score_max"]
    )

    return merged_df.sort_values("final_score", ascending=False).head(1000)


def main():
    parser = argparse.ArgumentParser(description="Fuse supervised and unsupervised scores.")
    parser.add_argument(
        "--run_name",
        type=str,
        default="hysefus_run",
        help="A name for the final run file.",
    )
    args = parser.parse_args()

    supervised_dir = "outputs/extractor_predictions"
    unsupervised_dir = "outputs/symptom_sentences"
    output_dir = "runs"
    os.makedirs(output_dir, exist_ok=True)

    # Load supervised predictions
    sup_files = glob(os.path.join(supervised_dir, "*.csv"))
    df_supervised = pd.concat((pd.read_csv(f) for f in sup_files), ignore_index=True)

    # Load unsupervised predictions (from both models)
    df_unsup_nomic = pd.read_csv(
        os.path.join(unsupervised_dir, "nomic-embed-text-v1.5.csv")
    )
    df_unsup_modern = pd.read_csv(
        os.path.join(unsupervised_dir, "modernbert-embed-base.csv")
    )
    # Average the scores from the two unsupervised models
    df_unsupervised = pd.merge(
        df_unsup_nomic,
        df_unsup_modern,
        on=["doc_id", "symptom"],
        how="outer",
        suffixes=("_nomic", "_modern"),
    ).fillna(0)
    df_unsupervised["similarity"] = (
        df_unsupervised["similarity_nomic"] + df_unsupervised["similarity_modern"]
    ) / 2

    run_file_path = os.path.join(output_dir, f"{args.run_name}.txt")
    with open(run_file_path, "w") as f_out:
        for i in range(1, 22):
            symptom_id = f"query_{i}"

            # Prepare supervised scores for this symptom
            sup_symptom = df_supervised[["doc_id", symptom_id]].copy()
            sup_symptom = normalize(sup_symptom, symptom_id)

            # Prepare unsupervised scores for this symptom
            unsup_symptom = df_unsupervised[
                df_unsupervised["symptom"] == symptom_id
            ][["doc_id", "similarity"]].copy()
            unsup_symptom = unsup_symptom.rename(columns={"similarity": symptom_id})
            unsup_symptom = normalize(unsup_symptom, symptom_id)

            # Fuse
            fused_df = fuse_scores(sup_symptom, unsup_symptom, symptom_id=i)

            # Write to TREC run file
            for _, row in fused_df.iterrows():
                f_out.write(
                    f"{symptom_id} Q0 {row['doc_id']} 0 {row['final_score']:.6f} {args.run_name}\n"
                )

    print(f"Fusion complete. Run file saved to {run_file_path}")


if __name__ == "__main__":
    main()
