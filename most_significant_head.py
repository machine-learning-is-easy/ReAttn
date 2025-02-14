import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load Fill Mask Model (BERT)
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()


def get_most_significant_heads(input_text):
    """Get the most significant attention head for each layer."""
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions  # List of (num_layers, batch, num_heads, seq_len, seq_len)

    significant_heads = []
    for layer_attention in attentions:
        avg_attention = layer_attention.mean(dim=(0, 2, 3))  # Average over batch and sequence length
        most_significant_head = torch.argmax(avg_attention).item()
        significant_heads.append(most_significant_head)

    return significant_heads


# Example usage
input_text = "I love [MASK] intelligence."
significant_heads = get_most_significant_heads(input_text)
print("Most significant heads per layer:", significant_heads)