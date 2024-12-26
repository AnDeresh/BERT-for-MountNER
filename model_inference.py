import torch

from scripts.inference_utils import *

def main():
    # Allow user input for text
    text = input("Enter the text for mountain name prediction: ")

    # Specify the paths to the model and tokenizer
    model_path = './model'  # Replace with your model's path
    tokenizer_path = './tokenizer'  # Replace with your tokenizer's path
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path, tokenizer_path)
    
    # Set up device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Run prediction
    predicted_mountains = predict(text, model, tokenizer, device)
    
    # Print the predicted mountain names
    if predicted_mountains:
        print("Predicted Mountain Names:", predicted_mountains)
    else:
        print("No mountains detected in the text.")

if __name__ == '__main__':
    main()
