import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load IMDB dataset (or any dataset)
dataset = load_dataset("imdb")

# Load the pre-trained LLaMA model and tokenizer
MODEL_NAME = "decapoda-research/llama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# Tokenize the dataset (no need for training)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create a DataLoader for evaluation
from torch.utils.data import DataLoader

# Prepare the dataset for evaluation
eval_dataset = tokenized_datasets["test"].remove_columns(["label", "text"])  # remove unwanted columns
eval_loader = DataLoader(eval_dataset, batch_size=8)


# Evaluate the model on the dataset (no training)
def evaluate_model(model, eval_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


# Evaluate the model
accuracy = evaluate_model(model, eval_loader)
print(f"Accuracy: {accuracy * 100:.2f}%")