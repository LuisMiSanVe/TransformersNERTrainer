import json
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline)

# Load trained model for inference
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=7)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

ner_pipeline = pipeline("ner", model="./my_ner_model", tokenizer=tokenizer, aggregation_strategy="first")

# Perform inference
text = "Text with data that matches the NER Model's training dataset."
tokens = tokenizer(text, return_tensors="pt", truncation=True)

# Ensure the model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokens = {k: v.to(device) for k, v in tokens.items()}

# Get predictions
with torch.no_grad():
    outputs = model(**tokens)

# Process the predictions
predictions = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
logits = outputs.logits[0]
predicted_labels = torch.argmax(logits, dim=-1)

# Convert numeric labels to meaningful names
# Change LabelX (NOT LABEL_X) with the actual fields the NER Model was trained with
label_map = {
    "LABEL_0": "O", 
    "LABEL_1": "B-LABEL1", 
    "LABEL_2": "I-LABEL1", 
    "LABEL_3": "B-LABEL2",
    "LABEL_4": "I-LABEL2",
    "LABEL_5": "B-LABEL3",
    "LABEL_6": "I-LABEL3",
}

# Format the output properly
final_predictions = []
for token, label in zip(predictions, predicted_labels):
    entity_label = f"LABEL_{label.item()}"  # Convert index to label format
    readable_label = label_map.get(entity_label, "UNKNOWN")  # Map label name
    final_predictions.append({"word": token, "entity": readable_label})

# Print readable predictions
# Filter out unwanted tokens like <s> or </s>
final_predictions = [pred for pred in final_predictions if pred["word"] not in ["<s>", "</s>"]]
print("Prediction:\n" + str(final_predictions))