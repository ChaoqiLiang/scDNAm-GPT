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
from src.model.scwgbs_gpt import scWGBSGPTForSequenceClassification, scWGBSGPTForSequenceClassificationWithBatchCorrection
from src.dataset.scwgbs_dataset import TokensRatiosDataset, scWGBS_collate_TokensRatios
from sklearn import metrics
from safetensors.torch import load_file as safetensors_load
import torch.distributed as dist
import uuid

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
            
            # Construct path for both formats
            best_model_path_pytorch = os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin")
            best_model_path_safetensors = os.path.join(self.state.best_model_checkpoint, "adapter_model.safetensors")
            
            # Try loading from safetensors first (if the library is available)
            if os.path.exists(best_model_path_safetensors):
                try:
                    state_dict = safetensors_load(best_model_path_safetensors)  # Load using safetensors
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded best model from safetensors: {best_model_path_safetensors}")
                except Exception as e:
                    print(f"Failed to load model from safetensors: {e}. Trying pytorch_model.bin...")
                    if os.path.exists(best_model_path_pytorch):
                        state_dict = torch.load(best_model_path_pytorch, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                        print(f"Loaded best model from pytorch_model.bin: {best_model_path_pytorch}")
                    else:
                        print(f"Best model not found at {best_model_path_pytorch}. Loading standard model.")
            elif os.path.exists(best_model_path_pytorch):
                # Fallback to pytorch_model.bin if safetensors is not available or fails
                state_dict = torch.load(best_model_path_pytorch, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded best model from pytorch_model.bin: {best_model_path_pytorch}")
            else:
                print(f"Best model not found at {best_model_path_pytorch} or {best_model_path_safetensors}. Loading standard model.")

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
    
    has_batch = isinstance(all_logits, tuple) # Handle tuple logits
    if has_batch:
        logits = all_logits[0]
        batch_logits = all_logits[1]
        batch_reversed_logits = all_logits[2]
        labels = all_labels[0]
        batches = all_labels[1]
    else:
        logits = all_logits
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
        "recall": metrics.recall_score(labels, predictions, average="macro", zero_division=0),
        # "batch_accuracy": metrics.accuracy_score(batches, batch_predictions),
        # "batch_matthews_correlation": metrics.matthews_corrcoef(batches, batch_predictions),
    }

    if has_batch:
        re_dict["batch_accuracy"] = metrics.accuracy_score(batches, batch_predictions)
        re_dict["batch_matthews_correlation"] = metrics.matthews_corrcoef(batches, batch_predictions)

    return re_dict

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
    train_dataset = TokensRatiosDataset(training_args_dict["train_csv_file"],
                                        training_args_dict["train_root_path"], 
                                        K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                        batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                        need_labels=True, need_batch=training_args_dict.get("need_batch", False),
                                        max_length=training_args_dict.get("max_length", None), 
                                        random=training_args_dict.get("train_random"),
                                        use_sample=training_args_dict.get("use_sample", False),
                                        use_truncation=training_args_dict.get("use_truncation", True), 
                                        start_idx=0, selective_chrs=training_args_dict.get("selective_chrs", None))
    
    val_dataset = TokensRatiosDataset(training_args_dict["val_csv_file"],
                                      training_args_dict["val_root_path"], 
                                      K_mer, tokenizer, type_json_path=training_args_dict["type_json_path"], 
                                      batch_type_json_path=training_args_dict.get("batch_type_json_path", None),
                                      need_labels=True, need_batch=training_args_dict.get("need_batch", False),
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
                                       max_length=training_args_dict.get("max_length", None), 
                                       random=training_args_dict.get("test_random"),
                                       use_sample=training_args_dict.get("use_sample", False),
                                       use_truncation=training_args_dict.get("use_truncation", True), 
                                       start_idx=0, selective_chrs=training_args_dict.get("selective_chrs", None))
    
    data_collator = scWGBS_collate_TokensRatios(tokenizer=tokenizer)

    # Initialize model
    Classifier = scWGBSGPTForSequenceClassificationWithBatchCorrection if training_args_dict.get("need_batch", False) else scWGBSGPTForSequenceClassification
    model = Classifier.from_pretrained(
        tokenizer=tokenizer,
        pretrained_model_name=training_args_dict["pretrained_model_path"],
        num_labels=train_dataset.num_labels,  # Assuming celltype is the label column
        attention_mechanism=training_args_dict["attention_mechanism"],
        batch_correction=training_args_dict.get("need_batch", False),
        lambd=training_args_dict.get("lambd", 1.0),
        num_batches=train_dataset.num_batches
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
        trainer_class=scWGBSTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=data_collator,
        training_args=training_args,
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
    rank_str = os.environ.get("SLURM_PROCID", "0")
    unique_id = str(uuid.uuid4())  # Generate a unique ID
    os.environ["TRITON_CACHE_DIR"] = f"triton_cache/finetuning/{unique_id}_{rank_str}"
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
    main()
