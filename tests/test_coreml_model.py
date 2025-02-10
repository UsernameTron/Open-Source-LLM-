import coremltools as ct
from transformers import DistilBertTokenizer
import numpy as np

def load_model():
    """Load the CoreML model."""
    model = ct.models.MLModel('models/sentiment_model_metal_metal.mlpackage')
    return model

def prepare_input(text, tokenizer, max_length=512):
    """Prepare input text for the model."""
    # Tokenize the input text
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    return {
        'input_ids': inputs['input_ids'].astype(np.int32),
        'attention_mask': inputs['attention_mask'].astype(np.int32)
    }

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for the given text."""
    # Prepare input
    inputs = prepare_input(text, tokenizer)
    
    # Make prediction
    prediction = model.predict(inputs)
    
    # Get logits from the output
    logits = prediction['linear_37']
    print(f"\nRaw logits: {logits}")
    
    # Convert logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    print(f"Probabilities: {probs}")
    
    # Get predicted class (0: negative, 1: positive)
    predicted_class = np.argmax(probs)
    confidence = float(probs[0][predicted_class])
    
    return {
        'sentiment': 'positive' if predicted_class == 1 else 'negative',
        'confidence': confidence
    }

def main():
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = load_model()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Test cases
    test_texts = [
        "I absolutely loved this movie! The acting was fantastic.",
        "This product is terrible, I want my money back.",
        "The service was okay, nothing special but not bad either.",
        "This is the best purchase I've ever made!"
    ]
    
    print("\nRunning sentiment analysis on test cases:")
    print("-" * 50)
    
    for text in test_texts:
        result = predict_sentiment(text, model, tokenizer)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")

if __name__ == '__main__':
    main()
