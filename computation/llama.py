from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer, DataCollatorForSeq2Seq
from datetime import datetime
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
import sys

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

torch.cuda.empty_cache()
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
new_model = "Llama-3.2-1B/adapter_model"
model_id = "meta-llama/Llama-3.2-1B"
# model_id = "meta-llama/Llama-3.2-11B-Vision"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="model_cache/", quantization_config=quantization_config)

model.train() # put model back into training mode
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# data loading

from datasets import load_dataset
dataset = load_dataset("b-mc2/sql-create-context", split="train", cache_dir='dataset')
train_dataset = dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="model_cache/")
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""You are a robust text-to-SQL model responsible for answering database-related questions. 
    You will receive a question and relevant context from one or more tables. 
    Your task is to create the SQL query that provides the answer..
        ### Input:
        {data_point["question"]}
        ### Context:
        {data_point["context"]}
        ### Response:
        {data_point["answer"]}
        """
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

batch_size = 16
per_device_train_batch_size = 2
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "sql-code-llama"

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=2,
        # max_steps=40000,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=100,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)
trainer.train()
trainer.save_model("save_model")

# model.train()

print("Training is complete")
# test the model
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

base_model = "sql-code-llama"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
from peft import PeftModel
model = PeftModel.from_pretrained(model, output_dir)
eval_prompt = """You are a robust text-to-SQL model responsible for answering database-related questions. 
    You will receive a question and relevant context from one or more tables. 
    Your task is to create the SQL query that provides the answer.
    ### Input:
    Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?
    ### Context:
    CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)
    ### Response:
    """

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

