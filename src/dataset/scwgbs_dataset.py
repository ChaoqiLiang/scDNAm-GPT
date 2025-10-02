import numpy as np
import pandas as pd
import torch, json, random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Sequence
from dataclasses import dataclass
import ast


# Define a global K-mer configuration dictionary
K_mer_dict = {
    "8mer": {"N_mod": 8192, "seq_len": 425000},
    "6mer": {"N_mod": 512, "seq_len": 2000000},
    "4mer": {"N_mod": 32, "seq_len": 2000000},
    "2mer": {"N_mod": 2, "seq_len": 2000000},
}


class TokensRatiosDataset(Dataset):
    """
    A PyTorch Dataset for handling tokenized sequences and methylation ratios.

    Args:
        csv_file (str): Path to the CSV file containing file information.
        root_path (str): Root directory where data files are stored.
        K_mer (str): The K-mer type, must be a key in `K_mer_dict`.
    """
    def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
                 type_json_path: str = None, batch_type_json_path: str = None,
                 need_labels: bool = False, need_batch: bool = False,
                 need_analysis=False, use_length_scale: bool = False, bias_power: float=0.4, min_length: int = 50000, max_length: int = None,
                 random: bool = False, use_truncation: bool = False, use_sample: bool = False,
                 start_idx: int = 0, selective_chrs: list = None):
        self.file_info = pd.read_csv(csv_file)
        # self.file_info = self.file_info.sample(frac=1).reset_index(drop=True)
        self.root_path = Path(root_path)
        self.need_labels = need_labels
        self.use_length_scale = use_length_scale
        self.bias_power = bias_power
        self.min_length = min_length
        self.max_length = max_length
        self.use_truncation = use_truncation
        self.use_sample = use_sample
        self.selective_chrs = selective_chrs
        self.random = random
        self.start_idx = start_idx
        if K_mer not in K_mer_dict:
            raise ValueError(f"Invalid K-mer type. Available options are: {list(K_mer_dict.keys())}")
        self.N_mod = K_mer_dict[K_mer]["N_mod"]
        if self.max_length == None:
            self.max_length = K_mer_dict[K_mer]["seq_len"]
        # Check and set whether 'celltype' and 'batch' are present
        if need_labels and 'celltype' not in self.file_info.columns:
            # Raise an error if 'need_labels' is True but 'celltype' column is missing
            raise ValueError("The column 'celltype' is required in file_info but is missing.")
        self.has_celltype = 'celltype' in self.file_info.columns and need_labels

        if need_batch and 'batch' not in self.file_info.columns:
            # Raise an error if 'need_batch' is True but 'batch' column is missing
            raise ValueError("The column 'batch' is required in file_info but is missing.")
        self.has_batch = 'batch' in self.file_info.columns and need_batch

        self.need_analysis = need_analysis
        if self.has_celltype:
            with open(type_json_path, "r") as file:
                self.label_to_id = json.load(file)
                self.num_labels = len(set(self.label_to_id.values()))
                self.file_info = self.file_info[self.file_info['celltype'].isin(list(self.label_to_id.keys()))]

        if self.has_batch:
            with open(batch_type_json_path, "r") as file:
                self.batch_to_id = json.load(file)
                self.num_batches = len(self.batch_to_id)
        else:
            self.num_batches = None
                
        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.end_token_id = tokenizer._convert_token_to_id("[SEP]")

    def loadata(self, file_path):
        try:
            # Ensure file_path is a Path object
            file_path = Path(file_path)
            
            if file_path.suffix == '.npy':  # Check for .npy extension
                return np.load(file_path)
            elif file_path.suffix == '.npz':  # Check for .npz extension
                return np.load(file_path)["data"]
            else:
                print(f"Warning: File path {file_path} is not a .npy or .npz file. Returning an empty array.")
                return np.array([])
        except Exception as e:
            print(f"Error while loading file {file_path}: {e}. Returning an empty array.")
            return np.array([])

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.file_info)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a single sample with either random or fixed sampling based on self.random_sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys `unmethy_input_ids`, `methy_ratios`, and optionally `labels`.
        """
        token_path = self.root_path / self.file_info.iloc[idx]['file_name']
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")

        tokens_data = self.loadata(token_path)
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)
        
        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
        ratios_path = Path(ratios_path)
        if not ratios_path.exists():
            raise FileNotFoundError(f"Ratios file not found: {ratios_path}")

        methy_ratios = self.loadata(ratios_path)

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)
        
        if self.need_analysis or self.selective_chrs is not None:
            if self.need_analysis:
                positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
                positions_path = Path(positions_path)
                if not positions_path.exists():
                    raise FileNotFoundError(f"Positions file not found: {positions_path}")
                positions = self.loadata(positions_path)
            
            chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
            chrs_path = Path(chrs_path)
            if not chrs_path.exists():
                raise FileNotFoundError(f"Chrs file not found: {chrs_path}")
            chrs = self.loadata(chrs_path)
            
            if self.selective_chrs is not None:
                filter_chrs_idx = np.isin(chrs, self.selective_chrs)

                valid_mask_chrs = valid_mask & filter_chrs_idx

                unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
                methy_input_ids = methy_input_ids[valid_mask_chrs]
                methy_ratios = methy_ratios[valid_mask_chrs]
                if self.need_analysis:
                    positions = positions[valid_mask_chrs]
                chrs = chrs[valid_mask_chrs]
        else:
            # Filter all the arrays using the valid_mask
            unmethy_input_ids = unmethy_input_ids[valid_mask]
            methy_input_ids = methy_input_ids[valid_mask]
            methy_ratios = methy_ratios[valid_mask]

        length = len(unmethy_input_ids)

        if self.need_labels and self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        if length >= max_length:
            if self.use_sample:
                if self.random:
                    # Random sampling
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    # Fixed sampling: evenly spaced indices
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    # Random truncation: Select a contiguous range of max_length
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            if self.need_analysis or self.selective_chrs is not None:
                if self.need_analysis:
                    positions = positions[sample_indices]
                chrs = chrs[sample_indices]

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
        }
        
        if self.need_analysis or self.selective_chrs is not None:
            if self.need_analysis:
                positions = np.concatenate(([0], positions, [0]))
                sample["positions"] = positions
            chrs = np.concatenate(([0], chrs, [0]))
            sample["chrs"] = chrs
            sample["file_names"] = self.file_info.iloc[idx]['file_name']

        if self.has_celltype:
            sample["labels"] = self.label_to_id[self.file_info.iloc[idx]['celltype']]

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample


@dataclass
class scWGBS_collate_TokensRatios:
    """
    A PyTorch collate function for batching tokenized sequences and methylation ratios.

    Args:
        tokenizer: A tokenizer object with methods `_convert_token_to_id`.
        K_mer (str): The K-mer type, must match the dataset configuration.
    """
    tokenizer: object

    def __post_init__(self):
        self.pad_token_id = self.tokenizer._convert_token_to_id("[PAD]")

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        processed_unmethy_inputs, processed_methy_inputs, processed_methy_ratios, labels, batch_labels = [], [], [], [], []
        processed_positions, processed_chrs, file_names = [], [], []

        for item in instances:
            unmethy_input_ids = item['unmethy_input_ids']
            methy_input_ids = item['methy_input_ids']
            methy_ratios = item['methy_ratios']

            processed_unmethy_inputs.append(torch.tensor(unmethy_input_ids, dtype=torch.long))
            processed_methy_inputs.append(torch.tensor(methy_input_ids, dtype=torch.long))
            processed_methy_ratios.append(torch.tensor(methy_ratios, dtype=torch.float))

            if 'labels' in item:
                labels.append(item['labels'])

            if 'batch_labels' in item:
                batch_labels.append(item['batch_labels'])
                
            if 'positions' in item:
                processed_positions.append(torch.tensor(item['positions'], dtype=torch.long))

            if 'chrs' in item:
                processed_chrs.append(torch.tensor(item['chrs'], dtype=torch.long))
                
            if 'file_names' in item:
                file_names.append(item['file_names'])

        unmethy_inputs_padded = pad_sequence(processed_unmethy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_inputs_padded = pad_sequence(processed_methy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_ratios_padded = pad_sequence(processed_methy_ratios, batch_first=True, padding_value=-1.0)
        positions_padded = pad_sequence(processed_positions, batch_first=True, padding_value=-1) if processed_positions else None
        chrs_padded = pad_sequence(processed_chrs, batch_first=True, padding_value=-1) if processed_chrs else None
        file_names = file_names if file_names else None

        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else None
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long) if batch_labels else None

        return {
            'unmethy_input_ids': unmethy_inputs_padded,
            'methy_input_ids': methy_inputs_padded,
            'methy_ratios': methy_ratios_padded,
            'labels': labels_tensor,
            'batch_labels': batch_labels_tensor,
            "positions": positions_padded,
            "chrs": chrs_padded,
            "file_names": file_names,
        }




class TokensRatiosLoadALLDataset(TokensRatiosDataset):
    """
    A subclass of TokensRatiosDataset that loads all data at once during initialization.
    This class overrides the necessary methods to load data in bulk and access it directly.
    """
    def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
                 type_json_path: str = None, batch_type_json_path: str = None,
                 need_labels: bool = False, need_batch: bool = False,
                 need_analysis=False, use_length_scale: bool = False, bias_power: float=0.4, 
                 min_length: int = 50000, max_length: int = None,
                 random: bool = False, use_truncation: bool = False, use_sample: bool = False,
                 start_idx: int = 0, selective_chrs: list = None):
        # Initialize the parent class
        super().__init__(csv_file, root_path, K_mer, tokenizer, 
                         type_json_path, batch_type_json_path,
                         need_labels, need_batch,
                         need_analysis, use_length_scale, 
                         bias_power, min_length, max_length,
                         random, use_truncation, use_sample,
                         start_idx, selective_chrs)

        # Load all data at once in the initialization
        self.tokens_data = []
        self.methy_ratios_data = []
        self.chrs_data = []
        self.positions_data = [] if need_analysis else None
        self.files = self.file_info['file_name'].tolist()

        for file_name in self.files:
            token_path = self.root_path / file_name
            if token_path.exists():
                tokens = self.loadata(token_path)
                self.tokens_data.append(tokens)

                ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
                ratios_path = Path(ratios_path)
                if ratios_path.exists():
                    ratios = self.loadata(ratios_path)
                    self.methy_ratios_data.append(ratios)

                if need_analysis:
                    positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
                    positions_path = Path(positions_path)
                    if positions_path.exists():
                        positions = self.loadata(positions_path)
                        self.positions_data.append(positions)

                chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
                chrs_path = Path(chrs_path)
                if chrs_path.exists():
                    chrs = self.loadata(chrs_path)
                    self.chrs_data.append(chrs)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Override to directly access the pre-loaded data.
        """
        # Access the pre-loaded data
        tokens_data = self.tokens_data[idx]
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)

        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        methy_ratios = self.methy_ratios_data[idx]

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)

        # Apply the selective chromosome filtering if needed
        if self.need_analysis or self.selective_chrs is not None:
            if self.need_analysis:
                positions = self.positions_data[idx] if self.positions_data else np.array([])

            chrs = self.chrs_data[idx]
            
            if self.selective_chrs is not None:
                filter_chrs_idx = np.isin(chrs, self.selective_chrs)
                
                # Further filter the data based on chromosome selection
                valid_mask_chrs = valid_mask & filter_chrs_idx
                
                # Apply the combined mask for both NaN and selective chromosomes
                unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
                methy_input_ids = methy_input_ids[valid_mask_chrs]
                methy_ratios = methy_ratios[valid_mask_chrs]
                if self.need_analysis:
                    positions = positions[valid_mask_chrs]
                chrs = chrs[valid_mask_chrs]
        else:
            # Filter all the arrays using the valid_mask
            unmethy_input_ids = unmethy_input_ids[valid_mask]
            methy_input_ids = methy_input_ids[valid_mask]
            methy_ratios = methy_ratios[valid_mask]

        length = len(unmethy_input_ids)

        if self.need_labels and self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        # if self.need_labels and self.use_length_scale:
        #     if random.random() < 0.05:  # 5% chance
        #         max_length = self.max_length
        #     else:
        #         r = random.random() ** self.bias_power
        #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
        # else:
        #     max_length = self.max_length

        if length >= max_length:
            if self.use_sample:
                if self.random:
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            if self.need_analysis:
                positions = positions[sample_indices]
                chrs = chrs[sample_indices]

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
        }
        
        if self.need_analysis:
            if self.need_analysis:
                positions = np.concatenate(([0], positions, [0]))
                sample["positions"] = positions
            chrs = np.concatenate(([0], chrs, [0]))
            sample["chrs"] = chrs
            sample["file_names"] = self.files[idx]

        if self.has_celltype:
            sample["labels"] = self.label_to_id[self.file_info.iloc[idx]['celltype']]

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample
