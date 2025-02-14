import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

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


def masked_forward(input_text):
    """Perform inference with selected self-attention head active."""
    significant_head = get_significant_attention_heads(input_text)

    def custom_forward_hook(module, input, output):
        batch_size, num_heads, seq_len, _ = output.shape
        mask = torch.zeros_like(output)
        mask[:, significant_head, :, :] = 1
        return output * mask

    # Register forward hooks to mask self-attention heads
    hooks = []
    for layer in model.bert.encoder.layer:
        hooks.append(layer.attention.self.register_forward_hook(custom_forward_hook))

    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks after inference
    for hook in hooks:
        hook.remove()

    return outputs


# Example Inference
input_text = "I love [MASK] intelligence."
output = masked_forward(input_text)
print(tokenizer.decode(torch.argmax(output.logits, dim=-1)[0]))
