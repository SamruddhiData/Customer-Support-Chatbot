import json
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

# Load the dataset
with open("data/support_data.json.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# Load pretrained tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenization function
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./model",
    
    num_train_epochs=50,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete. Model saved in ./model")
