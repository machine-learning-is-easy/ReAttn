import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig
from computation.significance_head import CustomBertSelfAttention
# Load Fill Mask Model (BERT)
MODEL_NAME = "bert-base-uncased"
config = BertConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, config=config)
model.eval()


# Example Inference
input_text = "I love [MASK] intelligence."

num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads
head_mask = torch.ones(num_layers, num_heads)  # Start with all heads active
head_mask[0, 1] = 0  # Mask head index 1 in layer 0
head_mask[1, 2] = 0  # Mask head index 2 in layer 1

inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
# Expand head_mask to match batch size
head_mask = head_mask.unsqueeze(0)  # Add batch dimension

output = model(**inputs, head_mask=head_mask)
print(tokenizer.decode(torch.argmax(output.logits, dim=-1)[0]))