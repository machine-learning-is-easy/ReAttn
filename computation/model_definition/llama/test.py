from transformers import LlamaTokenizer

# Load model and tokenizer
model = CustomLlamaForSequenceClassification.from_pretrained("facebook/llama-7b")
tokenizer = LlamaTokenizer.from_pretrained("facebook/llama-7b")

# Sample input
inputs = tokenizer("This is a test sentence.", return_tensors="pt", padding=True, truncation=True)

# Generate a head mask to skip specific attention heads (e.g., skip heads 4, 5, and 6)
head_mask = torch.ones(model.config.num_attention_heads)  # Enable all heads
head_mask[4:7] = 0  # Mask heads 4, 5, and 6

# Forward pass with head mask
outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], head_mask=head_mask)

# Get model output (e.g., classification logits)
logits = outputs.logits