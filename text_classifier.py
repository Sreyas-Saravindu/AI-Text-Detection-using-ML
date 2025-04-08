import joblib
import re
import numpy as np
import sys

def preprocess_text(text):
    """
    Simple text preprocessing
    """
    # Convert to lowercase
    processed = text.lower()
    
    # Remove special characters
    processed = re.sub(r'[^\w\s]', ' ', processed)
    
    # Remove extra spaces
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    return processed

def load_model(model_path="logistic_regression_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    """
    Load the trained model and vectorizer
    """
    try:
        # Load model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print(f"Model loaded: {model_path}")
        print(f"Vectorizer loaded: {vectorizer_path}")
        
        return model, vectorizer
    
    except FileNotFoundError:
        print(f"Error: Could not find model or vectorizer file.")
        print(f"Make sure {model_path} and {vectorizer_path} exist.")
        return None, None
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def classify_text(text, model, vectorizer):
    """
    Classify a text as AI-generated or human-generated
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform text to features
    features = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Get confidence score
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction]
    else:
        # For models without predict_proba (like LinearSVC)
        decision_values = model.decision_function(features)[0]
        confidence = 1 / (1 + np.exp(-abs(decision_values)))
    
    # Return result
    result = {
        'prediction': 'AI-generated' if prediction == 1 else 'Human-generated',
        'confidence': float(confidence),
        'label': int(prediction)
    }
    
    return result

def main():
    """
    Main function for the text classifier
    """
    print("AI vs Human Text Classifier")
    print("===========================")
    
    # Load model and vectorizer
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        return
    
    # Check for command line input
    if len(sys.argv) > 1:
        # Text provided as command line argument
        text = ' '.join(sys.argv[1:])
        result = classify_text(text, model, vectorizer)
        
        print(f"\nInput Text: {text[:100]}..." if len(text) > 100 else f"\nInput Text: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
    else:
        # Interactive mode
        print("\nEnter text to classify (type 'exit' to quit):")
        
        while True:
            text = input("\n> ")
            
            if text.lower() == 'exit':
                break
            
            if not text.strip():
                continue
                
            result = classify_text(text, model, vectorizer)
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
    
    print("\nThank you for using the AI vs Human Text Classifier!")

if __name__ == "__main__":
    main()