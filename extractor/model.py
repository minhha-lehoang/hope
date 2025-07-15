import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_base_model(
    model_path="rafalposwiata/deproberta-large-depression", num_labels=21
):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    return base_model


def get_model(
    model_path="rafalposwiata/deproberta-large-depression",
    num_labels=21,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.1,
):
    base_model = get_base_model(model_path, num_labels)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_config).to(device)
    print(model.print_trainable_parameters())

    return model


def load_checkpoint(checkpoint_path):
    base_model = get_base_model()
    model = PeftModel.from_pretrained(
        base_model, checkpoint_path, device_map="auto"
    ).to(device)
    print(model.print_trainable_parameters())

    return model
