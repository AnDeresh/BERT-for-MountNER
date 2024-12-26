from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizer
import torch
from typing import Tuple, Dict, Any

from scripts.data_processing_utils import *  


def load_model(model_path: str, tokenizer_path: str) -> Tuple[Any, AutoTokenizer]:
    """
    Loads a pre-trained model and tokenizer for token classification from specified paths.

    Args:
        model_path (str): The path or model name for loading the pre-trained model.
        tokenizer_path (str): The path or model name for loading the pre-trained tokenizer.

    Returns:
        Tuple[Any, AutoTokenizer]: A tuple containing the loaded model and tokenizer.
        
    Raises:
        Exception: If an error occurs while loading the model or tokenizer.
    """
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise

def preprocess_text(text: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Preprocesses input text by tokenizing it and padding/truncating it to a maximum length of 512 tokens.

    Args:
        text (str): The input text to be tokenized.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the text.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing tokenized inputs (input_ids, attention_mask, etc.).
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs

def predict(text: str, model: Any, tokenizer: AutoTokenizer, device: torch.device) -> list:
    """
    Makes predictions for the input text using the pre-trained model and tokenizer.

    Args:
        text (str): The input text to make predictions on.
        model (Any): The pre-trained model used for token classification.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input text.
        device (torch.device): The device (CPU or GPU) to move the model and data to for inference.

    Returns:
        list: A list of predicted mountain names (or other named entities) extracted from the text.
    """
    # Preprocess the text
    inputs = preprocess_text(text, tokenizer)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during inference
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [model.config.id2label[label.item()] for label in predictions[0]]

    # Extract named entities (e.g., mountain names) from the tokens and labels
    mountain_names = extract_mountains(tokens, predicted_labels, tokenizer)
    return mountain_names

def extract_mountains(tokens: List[str], predicted_labels: List[str], tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Extracts mountain names from tokenized text based on predicted labels. The function identifies tokens
    that are part of a named entity (e.g., mountain names), merges subwords (e.g., "K" and "##2" to "K2"), 
    and excludes common words like "mount" and "mountain" unless they are part of a full name.

    Args:
        tokens (List[str]): A list of tokenized words.
        predicted_labels (List[str]): A list of predicted labels for each token (e.g., 'B-MOUNT', 'I-MOUNT', 'O').
        tokenizer (PreTrainedTokenizer): The tokenizer used to identify special tokens.

    Returns:
        List[str]: A list of extracted mountain names, with each name as a string.
    """
    current_mountain = []  # Holds the current mountain name being built
    mountain_names = []  # List of fully formed mountain names
    excluded_words = {'mount', 'mountain'}  # Exclude common terms unless part of a full name

    for token, label in zip(tokens, predicted_labels):
        # Skip special tokens like [SEP], [CLS], [PAD], [UNK]
        if token in tokenizer.all_special_tokens:
            continue

        # Handle subwords (combine subwords like 'K' and '##2' into 'K2' without spaces)
        if token.startswith('##'):
            token = token.replace('##', '')  # Merge subword token (e.g., '##2' to '2')

        # If the token is "Mount" or "Mountain", skip it unless it's part of a full name
        if token.lower() in excluded_words:
            if current_mountain:
                # Only append if we have a full mountain name (not just "mount" or "mountain")
                mountain_name = "".join(current_mountain)
                if mountain_name.lower() not in excluded_words:
                    mountain_names.append(mountain_name)
            current_mountain = []  # Skip adding this token
        elif label == 'B-MOUNT' or label == 'I-MOUNT':
            current_mountain.append(token)  # Add token to the current mountain name
        else:
            if current_mountain:
                # If a mountain name was being formed, add it
                mountain_name = "".join(current_mountain)
                if mountain_name.lower() not in excluded_words:
                    mountain_names.append(mountain_name)
            current_mountain = []  # Reset the current mountain name

    # If the last word(s) were part of a mountain name, add them
    if current_mountain:
        mountain_name = "".join(current_mountain)
        if mountain_name.lower() not in excluded_words:
            mountain_names.append(mountain_name)

    return mountain_names