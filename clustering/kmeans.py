import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from .embedding import read_data
from utils.constants import CHECKPOINT_PATH, RELEVANT_THRESHOLD

def load_avg_questionnaire_vectors(description_folder, model):
    """
    For each symptom, average all questionnaire vectors to get one vector per symptom.
    Returns:
        - symptom_to_vector: dict of symptom -> avg vector
    """
    symptom_to_vector = {}
    for fname in os.listdir(description_folder):
        if fname.endswith(".txt"):
            symptom = fname.replace(".txt", "")
            with open(os.path.join(description_folder, fname), "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                vectors = model.encode(lines)
                avg_vec = np.mean(vectors, axis=0)
                symptom_to_vector[symptom] = avg_vec
    return symptom_to_vector


def load_depression_model(model_checkpoint):
    """
    Load a fine-tuned depression model from checkpoint using PEFT
    Returns:
        - model: A SentenceTransformer-compatible wrapper for the loaded model
    """
    print(f"Loading depression model from checkpoint: {model_checkpoint}")
    
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "rafalposwiata/deproberta-large-depression",
        num_labels=21,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    
    # Load the fine-tuned model with PEFT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(
        base_model, model_checkpoint, device_map="auto"
    ).to(device)
    
    # Create a wrapper class to make it compatible with SentenceTransformer's encode method
    class ModelWrapper:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.model.eval()
        
        def encode(self, sentences, batch_size=8, show_progress_bar=False, device=None):
            if device is None:
                device = self.device
                
            # Handle both single sentence and list of sentences
            if isinstance(sentences, str):
                sentences = [sentences]
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                encoded = self.tokenizer(
                    batch, 
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    # Extract the final hidden state or logits as embeddings
                    # Using the logits as embeddings for simplicity
                    embeddings = outputs.logits.cpu().numpy()
                    all_embeddings.append(embeddings)
            
            # Concatenate all batch embeddings
            all_embeddings = np.vstack(all_embeddings)
            return all_embeddings
    
    # Return wrapped model
    wrapped_model = ModelWrapper(model, tokenizer, device)
    return wrapped_model


def find_symptom_sentences(
    data_path,
    description_folder,
    model_path,
    output_dir,
    topk=1000,
    use_depression_model=False,
    checkpoint_path=None,
):
    """
    For each symptom, find the topk most similar sentences based on embedding distance
    to the average of BDI-II questionnaire vectors.
    """
    # Load data
    df = read_data(data_path)
    df = df.rename(columns={"sentence": "doc_text", "doc_id": "docid"}, errors="ignore")
    print(f"Loaded {len(df)} sentences for symptom mapping.")

    # Load model
    if use_depression_model and checkpoint_path:
        model = load_depression_model(checkpoint_path)
        model_name = "deproberta_finetuned"
    else:
        model = SentenceTransformer(model_path, trust_remote_code=True)
        model_name = model_path.split("/")[-1]

    # Encode all sentences
    embeddings = model.encode(
        df["doc_text"].tolist(), show_progress_bar=True, batch_size=16, device="cuda"
    )

    # Load avg questionnaire vectors
    symptom_to_vector = load_avg_questionnaire_vectors(description_folder, model)

    # For each symptom, collect topk closest sentences
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for symptom, v in symptom_to_vector.items():
        dists = np.linalg.norm(embeddings - v, ord=2, axis=1)
        top_idx = np.argsort(dists)[:topk]
        rows = [df.iloc[idx] for idx in top_idx]
        out_df = pd.DataFrame(rows)
        out_df["dist_to_avg"] = dists[top_idx]
        out_df["similarity"] = 1 - (
            out_df["dist_to_avg"] / (out_df["dist_to_avg"].max() + 1e-9)
        )
        out_df[["docid", "doc_text", "similarity"]].to_csv(
            os.path.join(model_output_dir, f"{symptom}.csv"), index=False
        )
        print(f"Saved top {topk} for {symptom} using model {model_name}")


def main():
    """Main function to run the kmeans-like sentence collection."""
    data_path = "data/labels/relevant_predictions.csv"
    description_folder = "clustering/bdi-questionaire"
    output_dir = "outputs/symptom_sentences"

    # Run for nomic-embed-text-v1.5
    find_symptom_sentences(
        data_path,
        description_folder,
        model_path="nomic-ai/nomic-embed-text-v1.5",
        output_dir=output_dir,
        topk=5000,
    )

    # Run for modernbert-embed-base
    find_symptom_sentences(
        data_path,
        description_folder,
        model_path="nomic-ai/modernbert-embed-base",
        output_dir=output_dir,
        topk=5000,
    )


if __name__ == "__main__":
    main()
