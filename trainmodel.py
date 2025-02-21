import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline)

# Sample dataset
json_data = {
    "contents": [ # Change 'contents' to whatever the data type the NER Model will be trained
        {"label1": "iPhone 13", "label2": "Iphone 13 pro max 256 Gb RAM", "label3": "negro"}, # Change labelX to whatever field represents the element. You can add all the labels you want
        {"label1": "MacBook Pro", "label2": "Portátil MacBook extra light", "label3": "magenta"} # Add all the contents you want
    ]
}

# Convert JSON data to NER format
def create_ner_data(json_data):
    dataset = []
    label_map = {"O": 0, "B-LABEL1": 1, "I-LABEL1": 2, "B-LABEL2": 3, "I-LABEL2": 4, "B-LABEL3": 5, "I-LABEL3": 6}
    
    for item in json_data["contents"]:
        tokens = []
        labels = []
        
        for key, value in item.items():
            value = str(value)  # Convert value to string to avoid issues
            words = value.split()
            for i, word in enumerate(words):
                tokens.append(word)
                if key == "label1":
                    labels.append(label_map["B-LABEL1"] if i == 0 else label_map["I-LABEL1"])
                elif key == "label2":
                    labels.append(label_map["B-LABEL2"] if i == 0 else label_map["I-LABEL2"])
                elif key == "label3":
                    labels.append(label_map["B-LABEL3"] if i == 0 else label_map["I-LABEL3"])
                else:
                    labels.append(label_map["O"])
        
        dataset.append({"tokens": tokens, "ner_tags": labels})

    return dataset

# Prepare dataset
dataset = create_ner_data(json_data)
dataset = DatasetDict({"train": Dataset.from_list(dataset)})

# Load tokenizer
model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # Ignore special tokens
        elif word_idx != previous_word_idx:
            # Handle subwords properly
            labels.append(example["ner_tags"][word_idx])
        else:
            # Avoid assigning incorrect labels to subwords
            labels.append(-100)  # Ignore subword tokens for "O" or overlapping entities
        
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels)

# Load model
num_labels = 7  # Number of unique labels in label_map
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    processing_class=tokenizer,
)

# Train model
trainer.train()

# Save model
trainer.save_model("./my_ner_model")
tokenizer.save_pretrained("./my_ner_model")

from collections import Counter

# Check label distribution in the training data
label_counts = Counter()
for example in dataset["train"]:
    label_counts.update(example["ner_tags"])

print("Balance " + str(label_counts))