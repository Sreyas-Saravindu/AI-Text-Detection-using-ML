import torch

# Function to classify text with your trained BERT model
def classify_text(text, model, tokenizer, device):
    model.eval()  # Set model to evaluation mode
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get prediction and confidence
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()
    
    result = {
        'prediction': 'AI-generated' if prediction == 1 else 'Human-generated',
        'confidence': confidence,
        'label': prediction
    }
    
    return result

# Main function to handle user input
def main():
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    
    # Load the model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Try to load your fine-tuned model weights
    try:
        model.load_state_dict(torch.load('bert_model_weights.pt'))
        print("Successfully loaded fine-tuned model weights.")
    except:
        print("Could not load fine-tuned model weights. Using base BERT model.")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    print("\n===== AI Text Detector (BERT) =====")
    print("Enter text to classify or type 'exit' to quit")
    
    while True:
        # Get input from user
        print("\nEnter text to analyze:")
        user_input = input("> ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting program...")
            break
        
        # Skip if input is empty
        if not user_input.strip():
            print("Please enter some text to analyze.")
            continue
        
        # Classify the text
        try:
            result = classify_text(user_input, model, tokenizer, device)
            
            # Display results
            print("\nResults:")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
            
            # Add additional context based on confidence
            if result['confidence'] > 0.9:
                print("Verdict: High confidence in this prediction")
            elif result['confidence'] > 0.7:
                print("Verdict: Moderate confidence in this prediction")
            else:
                print("Verdict: Low confidence - this text has mixed characteristics")
                
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")

if __name__ == "__main__":
    main()