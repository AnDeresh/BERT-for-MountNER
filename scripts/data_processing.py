import random
import torch
from torch.utils.data import Dataset

# Define the augmentation function
def augment_data(mountain_list, base_sentences, num_augmentations=100):
    augmented_sentences = []
    augmented_labels = []

    for _ in range(num_augmentations):
        mountain = random.choice(mountain_list)  # Random mountain name
        sentence = random.choice(base_sentences).format(mountain)
        
        # Generate labels for multi-word entities
        tokens = mountain.split()
        labels = ["B-MOUNT"] + ["I-MOUNT"] * (len(tokens) - 1)
        
        sentence_tokens = sentence.split()
        sentence_labels = []
        for token in sentence_tokens:
            if token in tokens:
                index = tokens.index(token)
                sentence_labels.append(labels[index])
            else:
                sentence_labels.append("O")
        
        augmented_sentences.append(sentence)
        augmented_labels.append(sentence_labels)
    
    return augmented_sentences, augmented_labels

# Function to tokenize sentences and align labels
def tokenize_and_align_labels(sentences, annotations, tokenizer):

    tokenized_inputs = tokenizer(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        is_split_into_words=False
    )

    aligned_labels = []
    for i, label in enumerate(annotations):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:  # Special tokens ([CLS], [SEP], etc.)
                label_ids.append(-100)
            elif word_id < len(label):  # Ensure word_id is within bounds
                if word_id != previous_word_id:  # First token of a word
                    label_ids.append(label[word_id])  # Assign original label
                else:  # Subtokens
                    # Repeat the label for subwords
                    sub_label = "I-MOUNT" if label[previous_word_id] == "B-MOUNT" else label[previous_word_id]
                    label_ids.append(sub_label)
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    return tokenized_inputs, aligned_labels


# Define a custom dataset for token classification
class NERDataset(Dataset):
    def __init__(self, inputs, labels, label2id):
        self.inputs = inputs
        self.labels = labels
        self.label2id = label2id  # Map for converting labels to IDs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        label_ids = [
            self.label2id[label] if label != -100 else -100  # Convert label to ID, ignore -100
            for label in self.labels[idx]
        ]
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": torch.tensor(label_ids, dtype=torch.long)  # Create tensor from label IDs
        }