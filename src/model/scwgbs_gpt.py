from src.model.scwgbs_mixer_seq_simple import MambaLMHeadModel
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional
import sys, torch
from torch import nn
import numpy as np
from src.mambaconfig import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

sys.path.append("..")

import torch
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)

def random_mask(sequence, mask_token_id, mask_prob=0.1):
    """
    Randomly mask a sequence based on PyTorch operations, supports CUDA tensors.
    
    Args:
        sequence: Original sequence (Tensor)
        mask_token_id: Mask token ID (int)
        mask_prob: Mask probability (float)
        
    Returns:
        Masked sequence (Tensor)
    """
    # Generate random mask
    mask = torch.rand(sequence.shape, device=sequence.device) < mask_prob
    # Replace values based on mask
    return torch.where(mask, torch.tensor(mask_token_id, device=sequence.device), sequence)

def random_replace(sequence, tokenizer, replace_prob=0.1, replace_condition=None):
    """
    Randomly replace tokens in the sequence with random tokens that satisfy a condition.
    
    Args:
        sequence: Original sequence (Tensor), shape (batch_size, sequence_length).
        tokenizer: Tokenizer object used for token replacement.
        replace_prob: Probability of replacement (float).
        replace_condition: Custom replacement condition function (callable) to filter vocab tokens.
        
    Returns:
        Replaced sequence (Tensor).
    """
    # Get tokens that satisfy the condition
    vocab = torch.arange(tokenizer.vocab_size, device=sequence.device)
    if replace_condition:
        vocab = vocab[replace_condition(vocab)]
    
    # Generate Bernoulli distribution mask
    replace_mask = torch.bernoulli(torch.full(sequence.shape, replace_prob, device=sequence.device)).bool()

    # Randomly sample from the valid tokens
    random_tokens = vocab[torch.randint(len(vocab), sequence.shape, device=sequence.device)]

    # Replace tokens based on the mask
    return torch.where(replace_mask, random_tokens, sequence)

def random_replace_with_shared_mask(sequence1, sequence2, tokenizer, replace_prob=0.1, condition1=None):
    """
    Randomly replace tokens in both sequence1 and sequence2 with random tokens that satisfy a condition, 
    ensuring that replacement occurs at the same positions in both sequences.
    
    Args:
        sequence1: First sequence (Tensor).
        sequence2: Second sequence (Tensor).
        tokenizer: Tokenizer object used for token replacement.
        replace_prob: Probability of replacement (float).
        condition1: Replacement condition function for sequence1.
        
    Returns:
        (sequence1_replaced, sequence2_replaced): Replaced sequences.
    """
    # Generate shared replacement mask
    replace_mask = torch.bernoulli(torch.full(sequence1.shape, replace_prob, device=sequence1.device)).bool()

    # Get tokens that satisfy the condition for sequence1
    vocab = torch.arange(tokenizer.vocab_size, device=sequence1.device)
    if condition1:
        vocab1 = vocab[condition1(vocab)]
    else:
        vocab1 = vocab

    # Randomly sample tokens from vocab1 and vocab2 (for sequence2)
    random_tokens1 = vocab1[torch.randint(len(vocab1), sequence1.shape, device=sequence1.device)]
    random_tokens2 = random_tokens1 + 1  # Random tokens for sequence2 (can adjust logic here)

    # Replace tokens based on the shared mask
    sequence1_replaced = torch.where(replace_mask, random_tokens1, sequence1)
    sequence2_replaced = torch.where(replace_mask, random_tokens2, sequence2)

    return sequence1_replaced, sequence2_replaced

def random_noise(methy_ratios, scale=0.05):
    """
    Add random noise to methylation ratios, based on PyTorch operations.
    
    Args:
        methy_ratios: Original methylation ratios (Tensor)
        scale: Noise amplitude (float)
        
    Returns:
        Methylation ratios with added noise (Tensor)
    """
    # Generate normal distribution noise
    noise = torch.normal(mean=0.0, std=scale, size=methy_ratios.shape, device=methy_ratios.device)
    # Add noise and clamp the result to the range [0, 1]
    return torch.clamp(methy_ratios + noise, min=0.0, max=1.0)

class scWGBSGPTLMHeadModelwithLoss(MambaLMHeadModel):
    """
    A Language Model Head with additional support for loss calculations related to methylation.

    Attributes:
        pad_token_id: Token ID for the padding token.
        start_token_id: Token ID for the start token (e.g., [BOS]).
        end_token_id: Token ID for the end token (e.g., [SEP]).
    """
    def __init__(self, config, tokenizer, initializer_cfg=None, device=None, dtype=None, use_dataug=False):
        super(scWGBSGPTLMHeadModelwithLoss, self).__init__(config, initializer_cfg, device, dtype)

        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.end_token_id = tokenizer._convert_token_to_id("[SEP]")
        self.mask_token_id = tokenizer._convert_token_to_id("[MASK]")
        self.tokenizer = tokenizer
        self.use_dataug = use_dataug
        self.profile_ce_mem = True
        self.last_profile = {}

        # Define replacement conditions
        self.condition_unmethy = lambda x: (x > 6) & (x % 2 == 1)  # ID > 6 and odd
        # self.condition_methy = lambda x: (x > 6) & (x % 2 == 0)  # ID > 6 and even
        
        print("------------------------------------------\nShape of lm_head.weight: ", self.lm_head.weight.shape, "\n------------------------------------------")

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, **kwargs):
        """Load a pretrained model and configuration."""
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)
        if torch.cuda.device_count() == 1:
            device = "cuda:0"
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error: {e}")
            del state_dict['classify.weight']
            del state_dict['classify.bias']
            model.load_state_dict(state_dict, strict=False)
        return model

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary and handle unexpected or missing keys."""
        load_result = super().load_state_dict(state_dict, strict=strict)

        if strict:
            missing_keys = [key for key in self.state_dict().keys() if key not in state_dict]
            unexpected_keys = [key for key in state_dict.keys() if key not in self.state_dict().keys()]
            load_result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

        self.tie_weights()
        return load_result

class scWGBSGPTForSequenceClassification(scWGBSGPTLMHeadModelwithLoss):
    """
    A class for sequence classification based on methylation data.

    Attributes:
        device: The device where the model is deployed (CPU/GPU).
        num_labels: The number of classification labels.
        classify: Linear layer for classification.
        attention: Multihead attention mechanism.
        dropout: Dropout layer to prevent overfitting.
        norm: Layer normalization to stabilize training.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        need_layer_hidden_states: Optional[bool]=False,
        cross_attn_every_hidden_states: Optional[bool]=False,
        lambd=1.0,
        just_mamba: Optional[bool]=False,
        use_tumor_ratios: Optional[bool]=False
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, initializer_cfg, device, dtype)

        self.device = device
        self.num_labels = num_labels
        self.batch_correction = batch_correction
        self._keys_to_ignore_on_save = ['lm_head.weight']
        self.lambd = lambd
        self.need_layer_hidden_states = need_layer_hidden_states
        self.cross_attn_every_hidden_states = cross_attn_every_hidden_states
        self.just_mamba = just_mamba
        self.use_tumor_ratios = use_tumor_ratios

        # Configure attention dimensions
        self.projection_dim = attention_mechanism.get("projection_dim", None)
        self.attention_embed_dim = (
            self.projection_dim
            if isinstance(self.projection_dim, int) and self.projection_dim > 0 and not self.just_mamba
            else config.d_model
        )
        self.attention_num_heads = attention_mechanism["attention_num_heads"]
        self.dropout_rate = attention_mechanism.get("dropout_rate", 0.0)

        if not self.just_mamba:
            # Optional projection layers
            if self.projection_dim:
                if cross_attn_every_hidden_states:
                    # Create a list of projection layers for each backbone layer
                    self.query_proj_layers = nn.ModuleList(
                        [
                            nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                            for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                        ]
                    )

                    self.key_proj_layers = nn.ModuleList(
                        [
                            nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                            for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                        ]
                    )

                    self.value_proj_layers = nn.ModuleList(
                        [
                            nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                            for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                        ]
                    )
                else:
                    self.query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                    self.key_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                    self.value_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

                if batch_correction:
                    self.batch_query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

            if cross_attn_every_hidden_states:
                # Create a list of attention layers with the same length as backbone layers
                self.attention_layers = nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=self.attention_embed_dim,
                            num_heads=self.attention_num_heads,
                            batch_first=True,
                        )
                        for _ in range(len(self.backbone.layers) + 1)  # Number of attention layers = number of backbone layers
                    ]
                )
            else:
                # Multihead attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.attention_embed_dim,
                    num_heads=self.attention_num_heads,
                    batch_first=True,
                )

        # Regularization and classification layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm = nn.LayerNorm(self.attention_embed_dim)
        if not self.use_tumor_ratios:
            self.classify = nn.Linear(self.attention_embed_dim, num_labels, dtype=dtype).to(device)
            nn.init.xavier_uniform_(self.classify.weight)
            nn.init.zeros_(self.classify.bias)
        else:
            self.regression = nn.Linear(self.attention_embed_dim, 1, dtype=dtype).to(device)
            nn.init.xavier_uniform_(self.regression.weight)
            nn.init.zeros_(self.regression.bias)

        # Batch correction components
        if batch_correction:
            self.batch_norm = nn.LayerNorm(self.attention_embed_dim)
            self.batch_emd = nn.Embedding(1, config.d_model, dtype=dtype).to(device)
            self.batch_projection = nn.Linear(self.attention_embed_dim, num_batches, dtype=dtype).to(device)
            self.batch_attention = nn.MultiheadAttention(
                embed_dim=self.attention_embed_dim,
                num_heads=self.attention_num_heads,
                batch_first=True,
            )
            self.batch_classify = nn.Linear(self.attention_embed_dim, num_batches, dtype=dtype).to(device)
            nn.init.xavier_uniform_(self.batch_projection.weight)
            nn.init.zeros_(self.batch_projection.bias)

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, strict: bool = False, **kwargs):
        """
        Load a pretrained model with optional strictness.
        
        Args:
            tokenizer: The tokenizer to use.
            pretrained_model_name: Path or HuggingFace model name.
            device: Target device (cuda/cpu).
            dtype: Data type.
            strict: Whether to enforce exact parameter matching. Default is False.
            **kwargs: Other model init args.
        """
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)

        if torch.cuda.device_count() == 1:
            device = "cuda:0"
        if not torch.cuda.is_available():
            device = torch.device("cpu")

        # Load raw state dict
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)

        if not strict:
            # Filter out mismatched or unknown keys
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"[Skip] Shape mismatch for key: {k}, checkpoint: {v.shape}, model: {model_state_dict[k].shape}")
                else:
                    print(f"[Skip] Unexpected key not in model: {k}")
            state_dict = filtered_state_dict

        # Load weights
        load_result = model.load_state_dict(state_dict, strict=strict)

        # Print warnings if not strict
        if not strict:
            if load_result.missing_keys:
                print("[Missing keys]:")
                for k in load_result.missing_keys:
                    print(f"  - {k}")
            if load_result.unexpected_keys:
                print("[Unexpected keys]:")
                for k in load_result.unexpected_keys:
                    print(f"  - {k}")

        return model

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Retrieve the state dictionary, excluding unnecessary keys."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state.pop(prefix + 'lm_head.weight', None)
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary and handle unexpected or missing keys."""
        load_result = super().load_state_dict(state_dict, strict=strict)

        if strict:
            missing_keys = [key for key in self.state_dict().keys() if key not in state_dict]
            unexpected_keys = [key for key in state_dict.keys() if key not in self.state_dict().keys()]
            load_result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

        self.tie_weights()
        return load_result

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""

        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        if self.just_mamba:
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            norm_attn_outputs_for_save = cls_token_states
            norm_attn_outputs = self.norm(cls_token_states)
        
        else:
            if self.cross_attn_every_hidden_states:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = layer_hidden_states[0].size()

                # Initialize list to store attention outputs for each layer
                norm_attn_outputs, attn_weights = [], []
                
                # Iterate through each layer's hidden state and apply attention separately
                layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
                for i, hidden_states in enumerate(layer_hidden_states):

                    # Extract the CLS token states for each sequence and apply attention
                    cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                    key_padding_mask = ~attention_mask.bool()

                    if hasattr(self, "query_proj") or hasattr(self, "query_proj_layers"):
                        q_proj = self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                        k_proj = self.key_proj_layers[i](hidden_states)
                        v_proj = self.value_proj_layers[i](hidden_states)
                    else:
                        q_proj = cls_token_states.unsqueeze(1)
                        k_proj = hidden_states
                        v_proj = hidden_states
                    
                    # Apply attention independently for this layer
                    attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)

                    # Store the attention output for this layer
                    norm_attn_outputs.append(self.norm(attn_output))
                    attn_weights.append(attn_weight)
                
                if return_dict:
                    norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                    
                # Aggregate the attention outputs (e.g., sum or average across layers)
                norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
                attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
            
            else:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = hidden_states.size()
                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                key_padding_mask = ~attention_mask.bool()

                if hasattr(self, "query_proj"):
                    q_proj = self.query_proj(cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj(hidden_states)
                    v_proj = self.value_proj(hidden_states)
                else:
                    q_proj = cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states

                attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                attn_output = attn_output.squeeze(1)

                norm_attn_outputs = self.norm(attn_output)
                if return_dict:
                    norm_attn_outputs_for_save = norm_attn_outputs
        
        # Classify using the normalized attention output
        if not self.use_tumor_ratios:
            logits = self.classify(norm_attn_outputs)
        # print(logits)
        # Compute loss
            loss = nn.CrossEntropyLoss()(logits, labels)

            # Return the result based on return_dict flag
            if not return_dict:
                return (loss, logits) if loss is not None else (logits,)
            # print("!!!!!!!!!!!!!!!!!!", logits.shape)
            return {
                "logits": logits,
                "labels": labels,
                "loss": loss,
            }

        else:
            logits = self.regression(norm_attn_outputs)
            # Apply sigmoid to ensure the output is between 0 and 1
            predicted_probs = torch.sigmoid(logits).squeeze(dim=1)

            # Preprocess labels: divide by 100 and convert to float
            processed_labels = (labels.to(predicted_probs.dtype) / 100.0)

            # print("predicted_probs:", predicted_probs, "; processed_labels:", processed_labels)

            # Compute MSE loss
            loss = nn.MSELoss()(predicted_probs, processed_labels)

            # Return the result based on return_dict flag
            if not return_dict:
                return (loss, logits) if loss is not None else (logits,)
            
            return {
                "logits": logits,
                "labels": labels,
                "loss": loss,
                "predicted_probs": predicted_probs,
            }


class scWGBSGPTForSequenceClassificationWithBatchCorrection(scWGBSGPTForSequenceClassification):
    """
    A class for sequence classification based on methylation data.

    Attributes:
        device: The device where the model is deployed (CPU/GPU).
        num_labels: The number of classification labels.
        classify: Linear layer for classification.
        attention: Multihead attention mechanism.
        dropout: Dropout layer to prevent overfitting.
        norm: Layer normalization to stabilize training.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        lambd=1.0,
        *args,
        **kwargs
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, num_labels, attention_mechanism, initializer_cfg, device, dtype, batch_correction, lambd)

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        batch_labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            **kwargs,
        )

        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        last_non_padding_indices = attention_mask.sum(dim=1) - 1
        batch_size, seq_len, embed_dim = hidden_states.size()
        cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]

        batch_cls_token_states = self.batch_emd(torch.zeros((batch_size,), dtype=torch.int).to(cls_token_states.device))

        key_padding_mask = ~attention_mask.bool()

        if hasattr(self, "query_proj"):
            q_proj = self.query_proj(cls_token_states.unsqueeze(1))
            batch_q_proj = self.batch_query_proj(batch_cls_token_states.unsqueeze(1))
            k_proj = self.key_proj(hidden_states)
            v_proj = self.value_proj(hidden_states)
        else:
            q_proj = cls_token_states.unsqueeze(1)
            batch_q_proj = batch_cls_token_states.unsqueeze(1)
            k_proj = hidden_states
            v_proj = hidden_states

        attn_output, _ = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
        attn_output = attn_output.squeeze(1)

        batch_attn_output, _ = self.batch_attention(
            batch_q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask
        )
        batch_attn_output = batch_attn_output.squeeze(1)
        norm_batch_attn_outputs = self.batch_norm(batch_attn_output)
        batch_logits = self.batch_classify(norm_batch_attn_outputs)

        norm_attn_outputs = self.norm(attn_output) - norm_batch_attn_outputs

        logits = self.classify(norm_attn_outputs)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        batch_loss, batch_reversed_loss = None, None
        reversed_features = grad_reverse(norm_attn_outputs, lambd=1.0)
        batch_reversed_logits = self.batch_projection(reversed_features)
        batch_loss = nn.CrossEntropyLoss()(batch_logits, batch_labels)
        batch_reversed_loss = nn.CrossEntropyLoss()(batch_reversed_logits, batch_labels)

        loss += batch_loss + self.lambd * batch_reversed_loss

        logits = (logits, batch_logits, batch_reversed_logits)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutputWithPast(loss=loss, logits=logits)