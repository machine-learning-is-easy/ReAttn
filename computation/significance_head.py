

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention


class CustomBertSelfAttention(BertSelfAttention):
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        """Override self-attention to compute only the most significant head."""
        significant_head = self.significance_head  # Placeholder, should be handled per input
        batch_size, seq_length, hidden_size = hidden_states.size()
        num_attention_heads = self.num_attention_heads
        head_size = hidden_size // num_attention_heads

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Reshape to [batch, heads, seq_len, head_size]
        query_layer = mixed_query_layer.view(batch_size, seq_length, num_attention_heads, head_size).transpose(1, 2)
        key_layer = mixed_key_layer.view(batch_size, seq_length, num_attention_heads, head_size).transpose(1, 2)
        value_layer = mixed_value_layer.view(batch_size, seq_length, num_attention_heads, head_size).transpose(1, 2)

        # Only compute attention for the selected head
        query_layer = query_layer[:, significant_head, :, :].unsqueeze(1)
        key_layer = key_layer[:, significant_head, :, :].unsqueeze(1)
        value_layer = value_layer[:, significant_head, :, :].unsqueeze(1)

        # Perform scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (head_size ** 0.5)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_output = torch.matmul(attention_probs, value_layer)

        # Reshape back
        attention_output = attention_output.squeeze(1).contiguous().view(batch_size, seq_length,
                                                                         hidden_size // num_attention_heads)
        return attention_output
