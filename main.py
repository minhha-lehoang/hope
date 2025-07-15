import argparse
import os
from data_processing.trec_loader import main as trec_loader_main
from create_labeled_data import main as create_labeled_data_main
from extractor.infer import main as extractor_infer_main
from clustering.relevant_filter import filter_relevant_predictions
from clustering.embedding import main as embedding_main
from clustering.kmeans import main as kmeans_main
from clustering.similarity import main as similarity_main
from fusion import main as fusion_main
from evaluate import call_metrics


def main():
    parser = argparse.ArgumentParser(description="Run the Hysefus pipeline.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint for the extractor.",
    )
    parser.add_argument(
        "--gold_file",
        type=str,
        required=True,
        help="Path to the ground truth qrels file for evaluation.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="hysefus_run",
        help="A name for the final run file.",
    )
    args = parser.parse_args()

    print("--- Starting Hysefus Pipeline ---")

    # 1. Data Preparation
    print("\n--- Step 1: Data Preparation ---")
    if not os.path.exists("data/erisk25-t1-processed"):
        print("Running TREC loader...")
        trec_loader_main()
    else:
        print("Processed TREC data found, skipping TREC loader.")

    if not os.path.exists("data/labels/labeled_data_with_text.csv"):
        print("Creating labeled data...")
        create_labeled_data_main()
    else:
        print("Labeled data found, skipping creation.")

    # 2. Supervised Symptom Extraction
    print("\n--- Step 2: Supervised Symptom Extraction ---")
    extractor_output_dir = "outputs/extractor_predictions"
    if not os.path.exists(extractor_output_dir) or not os.listdir(
        extractor_output_dir
    ):
        print("Running supervised extractor...")
        extractor_infer_main(args.checkpoint_path)
    else:
        print("Extractor predictions found, skipping inference.")

    # 3. Unsupervised Semantic Clustering
    print("\n--- Step 3: Unsupervised Semantic Clustering ---")
    relevant_preds_file = "data/labels/relevant_predictions.csv"
    if not os.path.exists(relevant_preds_file):
        print("Filtering relevant predictions...")
        filter_relevant_predictions()
    else:
        print("Relevant predictions file found, skipping filter.")

    symptom_sentence_dir = "outputs/symptom_sentences"
    if not os.path.exists(symptom_sentence_dir) or not os.listdir(
        symptom_sentence_dir
    ):
        print("Running unsupervised symptom sentence mapping...")
        kmeans_main()
    else:
        print("Symptom sentences found, skipping mapping.")

    # 4. Score Fusion
    print("\n--- Step 4: Score Fusion ---")
    run_file_path = f"runs/{args.run_name}.txt"
    if not os.path.exists(run_file_path):
        print("Running score fusion...")
        fusion_main()
    else:
        print(f"Run file {run_file_path} found, skipping fusion.")

    # 5. Evaluation
    print("\n--- Step 5: Evaluation ---")
    if os.path.exists(run_file_path):
        print("Running evaluation...")
        # The evaluate script is now a command-line tool.
        # We can call it using os.system or subprocess, but for simplicity,
        # we will just print the command to run.
        print("\nTo evaluate the run, execute the following command:")
        print(
            f"python evaluate.py --gold_file {args.gold_file} --run_file {run_file_path}"
        )
    else:
        print("No run file found to evaluate.")

    print("\n--- Hysefus Pipeline Finished ---")


if __name__ == "__main__":
    main()
