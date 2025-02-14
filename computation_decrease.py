import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertSelfAttention

# Load Fill Mask Model (BERT)
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()


class RLModel(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(RLModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_heads)

    def forward(self, x):
        return self.fc(x)  # Output logits for each attention head


# Load trained RL model
num_heads = model.config.num_attention_heads
input_dim = 768  # BERT hidden size
rl_model = RLModel(input_dim, num_heads)
rl_model.load_state_dict(torch.load("rl_model.pth"))
rl_model.eval()


def get_significant_attention_heads(input_text):
    """Use RL model to get the most important self-attention head indices."""
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()(input_ids).mean(dim=1)
        attention_logits = rl_model(input_embeddings)
        significant_head = torch.argmax(attention_logits, dim=1).item()
    return significant_head


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


# Replace all self-attention layers with the custom one

def masked_forward(input_text):
    """Perform inference using only the most important self-attention head."""
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs


# Example Inference
input_text = "I love [MASK] intelligence."
head_significance = get_significant_attention_heads(input_text=input_text)
start = 0
step = 12
for layer in model.bert.encoder.layer:
    layer.attention.self = CustomBertSelfAttention(model.config)
    layer.attention.self.significance_head = head_significance[start:start + step]
    start += 12

output = masked_forward(input_text)
print(tokenizer.decode(torch.argmax(output.logits, dim=-1)[0]))
