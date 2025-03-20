from src.model.scwgbs_mixer_seq_simple import MambaLMHeadModel
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional
import sys, torch
from torch import nn
from src.mambaconfig import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import torch.nn.functional as F

sys.path.append("..")

@dataclass
class CausalLMOutputWithCustomLosses(CausalLMOutput):
    """
    Custom output for Causal Language Models that includes additional loss terms for methylated and unmethylated Cs.
    """
    loss_methylated_C: Optional[torch.FloatTensor] = None
    loss_unmethylated_C: Optional[torch.FloatTensor] = None
    loss_other: Optional[torch.FloatTensor] = None
    methylation_ratios_loss: Optional[torch.FloatTensor] = None

class scWGBSMambaLMHeadModelwithLoss(MambaLMHeadModel):
    """
    A Language Model Head with additional support for loss calculations related to methylation.

    Attributes:
        pad_token_id: Token ID for the padding token.
        start_token_id: Token ID for the start token (e.g., [BOS]).
        end_token_id: Token ID for the end token (e.g., [SEP]).
    """
    def __init__(self, config, tokenizer, initializer_cfg=None, device=None, dtype=None):
        super(scWGBSMambaLMHeadModelwithLoss, self).__init__(config, initializer_cfg, device, dtype)

        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.end_token_id = tokenizer._convert_token_to_id("[SEP]")
        print("------------------------------------------\nShape of lm_head.weight: ", self.lm_head.weight.shape, "\n------------------------------------------")

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs
    ):
        """Perform a forward pass of the model."""

        # Forward pass through the backbone
        hidden_states = self.backbone(
            (unmethy_input_ids, methy_input_ids, methy_ratios),
            inference_params=inference_params,
            **mixer_kwargs
        )

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        # Compute logits for language modeling
        lm_logits = self.lm_head(hidden_states)

        # Shift logits and labels for causal language modeling loss
        shift_logits = lm_logits[..., :-1, :].contiguous()
        unmethy_shift_labels = unmethy_input_ids[..., 1:].contiguous()
        methy_shift_labels = methy_input_ids[..., 1:].contiguous()
        methy_ratios_shift = methy_ratios[..., 1:].contiguous()

        # Masks for methylated/unmethylated C and other tokens
        mask_other = (unmethy_shift_labels != self.pad_token_id)

        # CrossEntropy Loss for classification
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        all_loss = (
            loss_fct(shift_logits.view(-1, shift_logits.size(-1)), unmethy_shift_labels.view(-1)) * (1.0 - methy_ratios_shift) +
            loss_fct(shift_logits.view(-1, shift_logits.size(-1)), methy_shift_labels.view(-1)) * methy_ratios_shift
        )

        # Calculate individual losses
        loss_other = all_loss[mask_other].mean() if mask_other.any() else torch.tensor(0.0, device=unmethy_shift_labels.device)

        # Average classification loss
        classification_loss = loss_other

        return CausalLMOutputWithCustomLosses(
            loss=classification_loss,
            logits=lm_logits,
            hidden_states=None,
            attentions=None,
            loss_methylated_C=None,
            loss_unmethylated_C=None,
            loss_other=loss_other,
            methylation_ratios_loss=None
        )

class scWGBSMamba6merForSequenceClassification(scWGBSMambaLMHeadModelwithLoss):
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
        attention_mechanism: dict=None,
        initializer_cfg=None,
        device=None,
        dtype=None
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, initializer_cfg, device, dtype)

        self.device = device
        self.num_labels = num_labels
        self._keys_to_ignore_on_save = ['lm_head.weight']
        
        # Attention dimensions
        self.projection_dim = attention_mechanism.get("projection_dim", None)
        self.attention_embed_dim = self.projection_dim if isinstance(self.projection_dim, int) and self.projection_dim > 0 else config.d_model
        self.attention_num_heads = attention_mechanism["attention_num_heads"]
        self.dropout_rate = attention_mechanism.get("dropout_rate", 0.0)
        
        # Optional projection layers
        if isinstance(self.projection_dim, int) and self.projection_dim > 0:
            self.query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
            self.key_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
            self.value_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=self.attention_embed_dim, num_heads=self.attention_num_heads)
        
        # Regularization and classification layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm = nn.LayerNorm(self.attention_embed_dim)
        self.classify = nn.Linear(self.attention_embed_dim, num_labels, dtype=dtype).to(device)
        nn.init.xavier_uniform_(self.classify.weight)
        nn.init.zeros_(self.classify.bias)

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, **kwargs):
        """Load a pretrained model and configuration."""
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        model.load_state_dict(state_dict, strict=False)
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
        **kwargs
    ):
        """Perform a forward pass of the model."""
        # Pass inputs through the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            **kwargs,
        )

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (methy_ratios != -100).int()

        # Process each sample in the batch
        avg_hidden_states = []
        for i in range(hidden_states.size(0)):
            # Extract CLS token and non-padding states
            last_non_padding_idx = (attention_mask[i].sum() - 1).item()
            cls_token_state = hidden_states[i, last_non_padding_idx:last_non_padding_idx + 1, :]
            non_padding_states = hidden_states[i, :last_non_padding_idx, :]

            # Apply projection layers if configured
            if hasattr(self, "query_proj"):
                q_proj = self.query_proj(cls_token_state)
                k_proj = self.key_proj(non_padding_states)
                v_proj = self.value_proj(non_padding_states)
            else:
                q_proj = cls_token_state
                k_proj = non_padding_states
                v_proj = non_padding_states

            # Compute cross-attention
            attn_output, _ = self.attention(q_proj, k_proj, v_proj)

            # Apply normalization and aggregate results
            attn_output = self.norm(attn_output)
            avg_hidden_states.append(attn_output.squeeze(0))

        # Combine representations across the batch
        avg_hidden_states = torch.stack(avg_hidden_states)
        logits = self.classify(avg_hidden_states)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        # Return results
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutputWithPast(loss=loss, logits=logits)
    
    
class scWGBSMamba6merForDeconvolution(scWGBSMambaLMHeadModelwithLoss):
    """
    A class for deconvolution based on methylation data.

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
        attention_mechanism: dict=None,
        initializer_cfg=None,
        device=None,
        dtype=None
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, initializer_cfg, device, dtype)

        self.device = device
        self.num_labels = num_labels
        self._keys_to_ignore_on_save = ['lm_head.weight']
        
        # Attention dimensions
        self.projection_dim = attention_mechanism.get("projection_dim", None)
        self.attention_embed_dim = self.projection_dim if isinstance(self.projection_dim, int) and self.projection_dim > 0 else config.d_model
        self.attention_num_heads = attention_mechanism["attention_num_heads"]
        self.dropout_rate = attention_mechanism.get("dropout_rate", 0.0)
        
        # Optional projection layers
        if isinstance(self.projection_dim, int) and self.projection_dim > 0:
            self.query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
            self.key_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
            self.value_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=self.attention_embed_dim, num_heads=self.attention_num_heads)
        
        # Regularization and classification layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm = nn.LayerNorm(self.attention_embed_dim)
        self.classify = nn.Linear(self.attention_embed_dim, num_labels, dtype=dtype).to(device)
        nn.init.xavier_uniform_(self.classify.weight)
        nn.init.zeros_(self.classify.bias)

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, **kwargs):
        """Load a pretrained model and configuration."""
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        model.load_state_dict(state_dict, strict=False)
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
        **kwargs
    ):
        """Perform a forward pass of the model."""
        # Pass inputs through the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            **kwargs,
        )

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (methy_ratios != -100).int()

        # Process each sample in the batch
        avg_hidden_states = []
        for i in range(hidden_states.size(0)):
            # Extract CLS token and non-padding states
            last_non_padding_idx = (attention_mask[i].sum() - 1).item()
            cls_token_state = hidden_states[i, last_non_padding_idx:last_non_padding_idx + 1, :]
            non_padding_states = hidden_states[i, :last_non_padding_idx, :]

            # Apply projection layers if configured
            if hasattr(self, "query_proj"):
                q_proj = self.query_proj(cls_token_state)
                k_proj = self.key_proj(non_padding_states)
                v_proj = self.value_proj(non_padding_states)
            else:
                q_proj = cls_token_state
                k_proj = non_padding_states
                v_proj = non_padding_states

            # Compute cross-attention
            attn_output, _ = self.attention(q_proj, k_proj, v_proj)

            # Apply normalization and aggregate results
            attn_output = self.norm(attn_output)
            avg_hidden_states.append(attn_output.squeeze(0))

        # Combine representations across the batch
        avg_hidden_states = torch.stack(avg_hidden_states)
        logits = self.classify(avg_hidden_states)
        
        # Predict the distribution of mixture
        preds = F.softmax(logits, dim=-1)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # loss = nn.CrossEntropyLoss()(logits, labels)
            loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits,dim=-1), labels)

        # Return results
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutputWithPast(loss=loss, logits=logits)