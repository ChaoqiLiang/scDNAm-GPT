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
from transformers import Trainer as scDNAmTrainer
from src.model.scdnam_gpt import scDNAmGPTForSequenceClassification, scDNAmGPTForSequenceClassificationWithBatchCorrection
from src.dataset.scdnam_dataset import TokensRatiosDataset, scDNAm_collate_TokensRatios, TokensRatiosLoadALLDataset
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
from safetensors.torch import load_file as safetensors_load
import torch.distributed as dist
import uuid
from src.mambaconfig import MambaConfig

def is_main_process():
    """
    Check if the current process is the main process in distributed training.
    
    Returns:
        bool: True if the process is the main process, False otherwise.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True  # Single-process or non-distributed mode
    return dist.get_rank() == 0

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


import os
import types
import torch
from peft import PeftModel, PeftConfig, get_peft_model
from safetensors.torch import load_file as safetensors_load


def initialize_trainer(trainer_class, model, tokenizer, train_dataset, val_dataset, data_collator, training_args, compute_metrics):
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    def save_model(self, output_dir: str = None, **kwargs):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            # PEFT-aware save
            if isinstance(self.model, PeftModel):
                print("[PEFT] Saving LoRA adapter...")
                self.model.save_pretrained(output_dir)
                # Save classifier manually if it exists
                if hasattr(self.model.model, "classify"):
                    torch.save(self.model.model.classify.state_dict(), os.path.join(output_dir, "classify_head.pt"))
                    print("[PEFT] Saved classifier head to classify_head.pt")
            else:
                print("[Base] Saving full model...")
                self.model.save_pretrained(output_dir)
        except Exception as e:
            print(f"[Warning] Failed to save the model: {e}")

        try:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"[Warning] Failed to save the tokenizer: {e}")

        try:
            if hasattr(self.model, "config") and self.model.config is not None:
                self.model.config.save_pretrained(output_dir)
        except Exception as e:
            print(f"[Warning] Failed to save the model config: {e}")


    # ========== Custom load_best_model ==========
    def custom_load_best_model(self):
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            print(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})")

            model_dir = self.state.best_model_checkpoint
            safetensors_path = os.path.join(model_dir, "adapter_model.safetensors")
            pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
            classify_path = os.path.join(model_dir, "classify_head.pt")

            try:
                if isinstance(self.model, PeftModel):
                    print("[PEFT] Loading adapter from best checkpoint...")
                    config = PeftConfig.from_pretrained(model_dir)
                    base_model = self.model.base_model.model  # Extract original base model
                    self.model = get_peft_model(base_model, config)
                    self.model.load_adapter(model_dir, adapter_name="default")

                    # Restore classifier head
                    if hasattr(self.model.model, "classify") and os.path.exists(classify_path):
                        self.model.model.classify.load_state_dict(torch.load(classify_path, map_location="cpu"))
                        print("[PEFT] Loaded classifier head from classify_head.pt")
                else:
                    print("[Base] Loading full model state_dict...")
                    if os.path.exists(safetensors_path):
                        state_dict = safetensors_load(safetensors_path)
                        self.model.load_state_dict(state_dict, strict=False)
                    elif os.path.exists(pytorch_path):
                        state_dict = torch.load(pytorch_path, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                    else:
                        print("Warning: No model weights found at checkpoint.")
            except Exception as e:
                print(f"Failed to load best model: {e}")

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
    all_logits, all_labels = eval_pred
    # print("!!!!!!!!!!!!!!!!!!!!!!!", all_logits, all_labels)
    has_batch = False #isinstance(all_logits, tuple) # Handle tuple logits
    if has_batch:
        logits = all_logits[0]
        batch_logits = all_logits[1]
        batch_reversed_logits = all_logits[2]
        labels = all_labels[0]
        batches = all_labels[1]
    else:
        logits = all_logits #[0]
        labels = all_labels

    # Compute probabilities and predictions

    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(probabilities, axis=-1)

    if has_batch:
        batch_reversed_probabilities = F.softmax(torch.tensor(batch_reversed_logits), dim=-1).numpy()
        batch_reversed_predictions = np.argmax(batch_reversed_probabilities, axis=-1)

        batch_probabilities = F.softmax(torch.tensor(batch_logits), dim=-1).numpy()
        batch_predictions = np.argmax(batch_probabilities, axis=-1)
    
    if is_main_process():
        print("Confusion Matrix:", metrics.confusion_matrix(labels, predictions))
        if has_batch:
            print("Batch-Reversed Confusion Matrix:", metrics.confusion_matrix(batches, batch_reversed_predictions))

    # Compute metrics
    re_dict = {
        "accuracy": metrics.accuracy_score(labels, predictions),
        "f1": metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": metrics.matthews_corrcoef(labels, predictions),
        "precision": metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": metrics.recall_score(labels, predictions, average="macro", zero_division=0)
    }

    if has_batch:
        re_dict["batch_accuracy"] = metrics.accuracy_score(batches, batch_predictions)
        re_dict["batch_matthews_correlation"] = metrics.matthews_corrcoef(batches, batch_predictions)

    return re_dict


# Metric computation for evaluation
def compute_regression_metrics(eval_pred):
    """
    Compute evaluation metrics for a probabilistic regression task.

    Args:
        eval_pred: Tuple of predicted logits (after sigmoid) and true labels.

    Returns:
        dict: Computed metrics (RMSE, PCC, Spearman, and other relevant metrics).
    """
    logits, labels = eval_pred

    # Apply sigmoid to get probabilities between 0 and 1
    predicted_probs = torch.sigmoid(torch.tensor(logits)).squeeze(dim=1).numpy()  # Convert to numpy for metrics computation
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    # Ensure that predicted_probs and labels are one-dimensional
    assert predicted_probs.shape == labels.shape, "Predicted probabilities and labels must have the same shape"

    # Compute RMSE (Root Mean Squared Error)
    rmse = np.sqrt(metrics.mean_squared_error(labels, predicted_probs))

    # Compute Pearson Correlation Coefficient (PCC)
    pcc, _ = pearsonr(labels, predicted_probs)

    # Compute Spearman's Rank Correlation (SPC)
    spc, _ = spearmanr(labels, predicted_probs)

    # Return computed metrics
    return {
        "rmse": rmse,
        "pcc": pcc,
        "spc": spc
    }


# Main function for finetuning
def main(training_args_dict):
    """
    Main function for fine-tuning the scDNAm model.
    """

    # Load model configurations
    model_config_dict = load_config(Path(training_args_dict["pretrained_model_path"]) / "config.json")

    # Load tokenizer
    K_mer = model_config_dict["K_mer"]
    tokenizer = AutoTokenizer.from_pretrained(f"src/tokenizers/scdnam_{K_mer}", trust_remote_code=True)

    # Load datasets and data collator
    # K_mer = args.tokenizer_config_path.split("_")[-1]
    train_dataset = TokensRatiosDataset(training_args_dict["train_csv_file"],
                                        training_args_dict["train_root_path"], 
                                        K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                        batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                        need_labels=True, need_batch=training_args_dict.get("need_batch", False),
                                        use_length_scale=training_args_dict.get("train_use_length_scale", True),
                                        bias_power=training_args_dict.get("train_bias_power", 0.4),
                                        min_length=training_args_dict.get("min_length", 50000),
                                        max_length=training_args_dict.get("max_length", 2000000), 
                                        random=training_args_dict.get("train_random"),
                                        use_sample=training_args_dict.get("use_sample", False),
                                        use_truncation=training_args_dict.get("use_truncation", True), 
                                        start_idx=0, selective_chrs=training_args_dict.get("selective_chrs", None))
    
    val_dataset = TokensRatiosDataset(training_args_dict["val_csv_file"],
                                      training_args_dict["val_root_path"], 
                                      K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                      batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                      need_labels=True, need_batch=training_args_dict.get("need_batch", False),
                                      use_length_scale=training_args_dict.get("val_use_length_scale", False),
                                      bias_power=training_args_dict.get("train_bias_power", None),
                                      min_length=training_args_dict.get("min_length", 50000),
                                      max_length=training_args_dict.get("max_length", None),
                                      random=training_args_dict.get("val_random"),
                                      use_sample=training_args_dict.get("use_sample", False),
                                      use_truncation=training_args_dict.get("use_truncation", True), 
                                      start_idx=0, selective_chrs=training_args_dict.get("selective_chrs", None))
    
    test_dataset = TokensRatiosDataset(training_args_dict["test_csv_file"],
                                       training_args_dict["test_root_path"], 
                                       K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                       batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                       need_labels=True, need_batch=training_args_dict.get("need_batch", False),
                                       use_length_scale=training_args_dict.get("test_use_length_scale", False),
                                       bias_power=training_args_dict.get("train_bias_power", None),
                                       min_length=training_args_dict.get("min_length", 50000),
                                       max_length=training_args_dict.get("max_length", None), 
                                       random=training_args_dict.get("test_random"),
                                       use_sample=training_args_dict.get("use_sample", False),
                                       use_truncation=training_args_dict.get("use_truncation", True), 
                                       start_idx=0, selective_chrs=training_args_dict.get("selective_chrs", None))
    
    data_collator = scDNAm_collate_TokensRatios(tokenizer=tokenizer)

    # If distributed training is being used
    if dist.is_available() and dist.is_initialized():
        local_rank = dist.get_rank()  # Get the rank of the current process
        device = torch.device(f"cuda:{local_rank}")  # Use the rank to assign the device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    Classifier = scDNAmGPTForSequenceClassificationWithBatchCorrection if training_args_dict.get("need_batch", False) else scDNAmGPTForSequenceClassification

    train_from_scratch = training_args_dict.get("train_from_scratch", False)
    use_tumor_ratios = training_args_dict.get("use_tumor_ratios", False)

    if train_from_scratch:
        # Initialize model randomly
        print("# Initialize model randomly")
        model = Classifier(
            tokenizer=tokenizer,
            config=MambaConfig(**model_config_dict),  # Initialize with the same configuration
            device=device,  # Ensure it's on the right device (GPU/CPU)
            num_labels=train_dataset.num_labels,  # Same number of labels for consistency
            attention_mechanism=training_args_dict["attention_mechanism"],
            batch_correction=training_args_dict.get("need_batch", False),
            cross_attn_every_hidden_states=training_args_dict.get("cross_attn_every_hidden_states", True),
            lambd=training_args_dict.get("lambd", 1.0),
            num_batches=train_dataset.num_batches,
            just_mamba=training_args_dict.get("just_mamba", False),
            use_tumor_ratios=use_tumor_ratios
        )
    else:
        print("# Load pretrained model")
        model = Classifier.from_pretrained(
            tokenizer=tokenizer,
            pretrained_model_name=training_args_dict["pretrained_model_path"],
            device=device,
            num_labels=train_dataset.num_labels,  # Assuming celltype is the label column
            attention_mechanism=training_args_dict["attention_mechanism"],
            batch_correction=training_args_dict.get("need_batch", False),
            cross_attn_every_hidden_states=training_args_dict.get("cross_attn_every_hidden_states", True),
            lambd=training_args_dict.get("lambd", 1.0),
            num_batches=train_dataset.num_batches,
            just_mamba=training_args_dict.get("just_mamba", False),
            use_tumor_ratios=use_tumor_ratios
        )

    if is_main_process():
        # Print which parameters are trainable
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"- {name}")

        # Count total and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
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
        if is_main_process():
            print("Lora fine-tuning initialized.")
            print(model)

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
        trainer_class=scDNAmTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=data_collator,
        training_args=training_args,
        compute_metrics=compute_metrics if not use_tumor_ratios else compute_regression_metrics
    )
    
    if is_main_process():
        # Save training arguments as JSON
        training_args_path = os.path.join(training_args_dict["training_args"]["logging_dir"], "training_args.json")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(training_args_path), exist_ok=True)
        with open(training_args_path, "w") as f:
            json.dump(training_args_dict, f, indent=4)
        print(f"Training arguments saved to {training_args_path}")

    # Train or resume training
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        if is_main_process():
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()

    if is_main_process():
        # Save model and tokenizer
        trainer.save_model(training_args_dict["training_args"]["logging_dir"])
        tokenizer.save_pretrained(training_args_dict["training_args"]["logging_dir"])

    # Evaluate on validation and test datasets
    eval_results = trainer.evaluate(val_dataset)
    if is_main_process():
        eval_results_path = os.path.join(training_args_dict["training_args"]["logging_dir"], "eval_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(eval_results, f)
        print("Validation results:", eval_results)

    test_results = trainer.evaluate(test_dataset)
    if is_main_process():
        test_results_path = os.path.join(training_args_dict["training_args"]["logging_dir"], "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_results, f)
        print("Test results:", test_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune a scDNAm model.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    # parser.add_argument("--tokenizer_config_path", type=str, required=True, help="Path to the model configuration JSON file.")
    parser.add_argument("--training_args_path", type=str, required=True, help="Path to the training arguments JSON file.")
    args = parser.parse_args()

    # 1) If MASTER_ADDR/MASTER_PORT are not set, infer from the Slurm node list
    #    Here, we use the first node as MASTER_ADDR, but you can modify this logic as needed
    if "MASTER_ADDR" not in os.environ:
        hostnames = os.popen("scontrol show hostnames $SLURM_JOB_NODELIST").read().split()
        master_name = hostnames[0] if hostnames else "localhost"
        os.environ["MASTER_ADDR"] = master_name

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "16866"  # Change to another port if needed

    # 2) Map Slurm variables to RANK / LOCAL_RANK / WORLD_SIZE for libraries that require them
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # 3) Read Slurm environment variables
    #    - SLURM_PROCID: global rank
    #    - SLURM_LOCALID: local rank within the node
    #    - SLURM_NTASKS: total number of tasks (world_size)
    if "SLURM_PROCID" in os.environ:
        global_rank = int(os.environ["SLURM_PROCID"])
    else:
        global_rank = os.environ["RANK"]
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 4) Bind the current process to the corresponding GPU (using local_rank as the index)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # 5) Set a unique Triton cache directory for each process to avoid conflicts during multi-process compilation
    training_args_dict = load_config(args.training_args_path)
    rank_str = os.environ.get("SLURM_PROCID", "0")
    unique_id = str(uuid.uuid4())  # Generate a unique ID
    os.environ["TRITON_CACHE_DIR"] = f"{training_args_dict['training_args']['logging_dir']}/triton_cache/finetuning/{unique_id}_{rank_str}"
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    # 6) Fix the random seed for reproducibility
    set_seed(42)

    # 7) Print debug information
    print(f"Hello from global rank {global_rank} (local rank={local_rank}),"
          f" MASTER_ADDR={os.environ.get('MASTER_ADDR')}," 
          f" MASTER_PORT={os.environ.get('MASTER_PORT')}")
    print("RANK =", os.environ.get("RANK"),
          "LOCAL_RANK =", os.environ.get("LOCAL_RANK"),
          "CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"),
          "GPU count (torch.cuda.device_count()) =", torch.cuda.device_count())

    # 8) Call your main logic
    main(training_args_dict)
