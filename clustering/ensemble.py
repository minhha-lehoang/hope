import os

import pandas as pd

from utils.constants import LIST_BDI_SYMPTOMS


def main(symptom_sentence_dir, output_path, id_column="doc_no"):
    """
    Average per-symptom similarity across embedding models (subdirs of
    symptom_sentence_dir), then merge into one wide CSV:
    `id` + `<symptom>_similarity` for every BDI symptom. Docs missing from a
    symptom's top-k get 0.
    """
    model_dirs = [
        os.path.join(symptom_sentence_dir, d)
        for d in sorted(os.listdir(symptom_sentence_dir))
        if os.path.isdir(os.path.join(symptom_sentence_dir, d))
    ]
    merged = None
    for symptom in LIST_BDI_SYMPTOMS:
        dfs = [pd.read_csv(os.path.join(d, f"{symptom}.csv")) for d in model_dirs]
        sim = (
            pd.concat(dfs)
            .groupby(id_column)["similarity"]
            .mean()
            .rename(f"{symptom}_similarity")
            .reset_index()
        )
        merged = sim if merged is None else merged.merge(sim, on=id_column, how="outer")

    merged = merged.fillna(0).rename(columns={id_column: "id"})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main("outputs/symptom_sentences", "outputs/unsupervised_preds.csv")
