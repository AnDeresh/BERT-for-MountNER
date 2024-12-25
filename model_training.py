import torch
import logging
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AdamW
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from scripts.data_processing import *

print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Set logging level to ERROR to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load the saved data
data = torch.load('preprocessed_data.pth')

train_inputs = data['train_inputs']
train_labels = data['train_labels']
test_inputs = data['test_inputs']
test_labels = data['test_labels']

# Define label-to-ID mapping
label2id = {"O": 0, "B-MOUNT": 1, "I-MOUNT": 2}

# Recreate the datasets
train_dataset = NERDataset(train_inputs, train_labels, label2id)
test_dataset = NERDataset(test_inputs, test_labels, label2id)

# Recreate the DataLoader objects
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Define the model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=3,
    id2label={0: "O", 1: "B-MOUNT", 2: "I-MOUNT"},
    label2id={"O": 0, "B-MOUNT": 1, "I-MOUNT": 2}
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class weights for imbalanced data
tag_counts = {'O': 10153, 'B-MOUNT': 82, 'I-MOUNT': 5}
total = sum(tag_counts.values())
weights = torch.tensor([
    total / tag_counts['O'],          # Weight for "O"
    total / tag_counts['B-MOUNT'],    # Weight for "B-MOUNT"
    total / tag_counts['I-MOUNT'],    # Weight for "I-MOUNT"
]).to(device)

# Use weighted loss
loss_function = CrossEntropyLoss(weight=weights)

# Prepare optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Move model to device (GPU if available)
model.to(device)

# List to store loss for each epoch
epoch_losses = []

# Training loop
model.train()
for epoch in range(50):  # Adjust the number of epochs
    print(f"Epoch {epoch + 1}")
    total_loss = 0

    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        # Compute loss
        loss = loss_function(logits.view(-1, model.config.num_labels), labels.view(-1))
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Store the average loss for the epoch
    average_loss = total_loss / len(train_dataloader)
    epoch_losses.append(average_loss)
    print(f"Average loss for epoch {epoch + 1}: {average_loss}")

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, len(epoch_losses) + 1))  # Set x-axis to show each epoch
plt.legend()
plt.grid()
plt.show()

# Define the folder where the model should be saved
save_folder = "./ner_model"

# Check if the folder exists
if not os.path.exists(save_folder):
    print(f"Creating folder: {save_folder}")
    os.makedirs(save_folder)
else:
    print(f"Saving model into existing folder: {save_folder}")

# Save the model
model.save_pretrained(save_folder)

print(f"Model and tokenizer saved in: {save_folder}")