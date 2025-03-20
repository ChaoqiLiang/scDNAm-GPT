import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from pathlib import Path
import json
import argparse
import torch
import types
import random
import numpy as np
import torch.nn.functional as F
from transformers import TrainingArguments, AutoTokenizer
from transformers import Trainer as scWGBSTrainer
from src.model.scwgbs_causal import scWGBSMamba6merForSequenceClassification, scWGBSMamba6merForDeconvolution
from src.dataset.scwgbs_dataset import TokensRatiosDataset, scWGBS_collate_TokensRatios_Deconv, TokensRatiosDeconvDataset
from sklearn import metrics
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import jensenshannon

# Fix random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.benchmark = False

# Function to load JSON configuration files
def load_config(json_path: str) -> dict:
    """
    Load a JSON configuration file.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(json_path, "r") as f:
        return json.load(f)


# Custom method for trainer initialization
def initialize_trainer(trainer_class, model, tokenizer, train_dataset, val_dataset, data_collator, training_args):
    """
    Initialize the Trainer with a custom save_model and _load_best_model method.

    Args:
        trainer_class: Trainer class to use.
        model: The model to train.
        tokenizer: The tokenizer instance.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        data_collator: Data collator for batching.
        training_args: Training arguments.

    Returns:
        scWGBSTrainer: Configured trainer instance.
    """
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Custom save_model method
    def save_model(self, output_dir: str = None, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        if hasattr(self.model, "config") and self.model.config is not None:
            self.model.config.save_pretrained(output_dir)

    # Custom method to load the best model
    def custom_load_best_model(self):
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            print(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
            best_model_path = os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin")
            if os.path.exists(best_model_path):
                state_dict = torch.load(best_model_path, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
            else:
                print(f"Best model not found at {best_model_path}. Loading standard model.")

    # Override Trainer methods
    trainer._load_best_model = types.MethodType(custom_load_best_model, trainer)
    trainer.save_model = types.MethodType(save_model, trainer)
    return trainer


# Metric computation for evaluation
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.

    Args:
        eval_pred: Tuple of logits and labels.

    Returns:
        dict: Computed metrics (accuracy, F1 score, MCC, precision, recall).
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Handle tuple logits
        logits = logits[0]

    # Compute probabilities and predictions
    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
    # predictions = np.argmax(probabilities, axis=-1)

    # # Filter out invalid labels
    # valid_mask = labels != -100
    # valid_predictions = predictions[valid_mask]
    # valid_labels = labels[valid_mask]
    
    # print("confusion_matrix", metrics.confusion_matrix(
    #         valid_labels, valid_predictions))

    # Compute metrics
    return {
        # "accuracy": metrics.accuracy_score(valid_labels, valid_predictions),
        # "f1": metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        # "matthews_correlation": metrics.matthews_corrcoef(valid_labels, valid_predictions),
        # "precision": metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        # "recall": metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "rmse": np.sqrt(metrics.mean_squared_error(labels, probabilities)),
        "pcc": np.nanmean(np.array([pearsonr(labels[i], probabilities[i])[0] for i in range(labels.shape[0])])), #pearsonr(labels.ravel(), probabilities.ravel())[0],
        "spc": np.nanmean(np.array([spearmanr(labels[i], probabilities[i])[0] for i in range(labels.shape[0])])), #spearmanr(labels.ravel(), probabilities.ravel())[0],
        "jsd": np.mean(list(map(lambda p, q: jensenshannon(p, q), probabilities / probabilities.sum(axis=1, keepdims=True), labels / labels.sum(axis=1, keepdims=True))))
        # "pcc": np.nanmean(pearsonr(labels, probabilities)[0])
    }


# Main function for finetuning
def main():
    """
    Main function for fine-tuning the scWGBS model.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a scWGBS model.")
    # parser.add_argument("--tokenizer_config_path", type=str, required=True, help="Path to the model configuration JSON file.")
    parser.add_argument("--training_args_path", type=str, required=True, help="Path to the training arguments JSON file.")
    args = parser.parse_args()

    # Load configurations
    training_args_dict = load_config(args.training_args_path)
    model_config_dict = load_config(Path(training_args_dict["pretrained_model_path"]) / "config.json")

    # Load tokenizer
    K_mer = model_config_dict["K_mer"]
    tokenizer = AutoTokenizer.from_pretrained(f"src/tokenizers/scwgbs_{K_mer}", trust_remote_code=True)

    # Load datasets and data collator
    # K_mer = args.tokenizer_config_path.split("_")[-1]
    # train_dataset = TokensRatiosDeconvDataset(training_args_dict["train_csv_file"],
    #                                     training_args_dict["train_root_path"], K_mer, finetuning=True)
    # val_dataset = TokensRatiosDeconvDataset(training_args_dict["val_csv_file"],
    #                                   training_args_dict["val_root_path"], K_mer, finetuning=True)
    # test_dataset = TokensRatiosDeconvDataset(training_args_dict["test_csv_file"],
    #                                    training_args_dict["test_root_path"], K_mer, finetuning=True)
    
    train_dataset = TokensRatiosDeconvDataset(training_args_dict["train_csv_file"],
                                        training_args_dict["train_root_path"], 
                                        K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                        batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                        need_labels=True, need_batch=training_args_dict.get("need_batch", False),
                                        max_length=training_args_dict.get("max_length", None), 
                                        random=training_args_dict.get("train_random"),
                                        use_sample=training_args_dict.get("use_sample", False),
                                        use_truncation=training_args_dict.get("use_truncation", True), start_idx=0)
    
    val_dataset = TokensRatiosDeconvDataset(training_args_dict["val_csv_file"],
                                      training_args_dict["val_root_path"], 
                                      K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                      batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                      need_labels=True, need_batch=training_args_dict.get("need_batch", False),
                                      max_length=training_args_dict.get("max_length", None),
                                      random=training_args_dict.get("val_random"),
                                      use_sample=training_args_dict.get("use_sample", False),
                                      use_truncation=training_args_dict.get("use_truncation", True), start_idx=0)
    
    test_dataset = TokensRatiosDeconvDataset(training_args_dict["test_csv_file"],
                                       training_args_dict["test_root_path"], 
                                       K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                       batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                       need_labels=True, need_batch=training_args_dict.get("need_batch", False),
                                       max_length=training_args_dict.get("max_length", None), 
                                       random=training_args_dict.get("test_random"),
                                       use_sample=training_args_dict.get("use_sample", False),
                                       use_truncation=training_args_dict.get("use_truncation", True), start_idx=0)
    
    data_collator = scWGBS_collate_TokensRatios_Deconv(tokenizer=tokenizer, K_mer=K_mer, max_length=training_args_dict.get("max_length", None))

    # Initialize model
    model = scWGBSMamba6merForDeconvolution.from_pretrained(
        tokenizer=tokenizer,
        pretrained_model_name=training_args_dict["pretrained_model_path"],
        num_labels=train_dataset.num_labels,  # Assuming celltype is the label column
        attention_mechanism=training_args_dict["attention_mechanism"]
    )
    
    # Check if Lora config exists and apply Lora fine-tuning
    lora_config = training_args_dict.get("lora_config", None)
    if lora_config:
        from peft import LoraConfig, get_peft_model
        lora_config_instance = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            task_type=lora_config["task_type"],
            target_modules=lora_config["target_modules"],
            modules_to_save=lora_config["modules_to_save"],
            inference_mode=lora_config["inference_mode"]
        )
        model = get_peft_model(model, lora_config_instance)
        print("Lora fine-tuning initialized.")

    # Initialize training arguments
    training_args = TrainingArguments(
        **training_args_dict["training_args"]
    )

    # Check for checkpoint in DeepSpeed configuration
    deepspeed_config_path = training_args_dict["training_args"].get("deepspeed", None)
    checkpoint_dir = None
    if deepspeed_config_path and os.path.exists(deepspeed_config_path):
        with open(deepspeed_config_path, "r") as ds_f:
            deepspeed_config = json.load(ds_f)
            checkpoint_dir = deepspeed_config.get("checkpoint", None)

    # Initialize trainer
    trainer = initialize_trainer(
        trainer_class=scWGBSTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=data_collator,
        training_args=training_args,
    )
    
    # Save training arguments as JSON
    training_args_path = os.path.join(training_args_dict["training_args"]["logging_dir"], "training_args.json")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(training_args_path), exist_ok=True)
    with open(training_args_path, "w") as f:
        json.dump(training_args_dict, f, indent=4)
    print(f"Training arguments saved to {training_args_path}")

    # Train or resume training
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()

    # Save model and tokenizer
    trainer.save_model(training_args_dict["training_args"]["logging_dir"])
    tokenizer.save_pretrained(training_args_dict["training_args"]["logging_dir"])

    # Evaluate on validation and test datasets
    eval_results = trainer.evaluate(val_dataset)
    eval_results_path = os.path.join(training_args_dict["training_args"]["logging_dir"], "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print("Validation results:", eval_results)

    test_results = trainer.evaluate(test_dataset)
    test_results_path = os.path.join(training_args_dict["training_args"]["logging_dir"], "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=4)
    print("Test results:", test_results)


if __name__ == "__main__":
    set_seed(42)
    main()
