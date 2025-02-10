import coremltools as ct
from transformers import AutoTokenizer
import numpy as np
import pprint
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

class SentimentAnalyzer:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = ct.models.MLModel('models/sentiment_model_metal.mlpackage')
        
        # Print model information
        print("\nModel Description:")
        print("=" * 50)
        print("Inputs:")
        pprint.pprint(self.model.input_description)
        print("\nOutputs:")
        pprint.pprint(self.model.output_description)
    
    def prepare_batch(self, sentences: List[str]) -> Dict:
        # Tokenize all sentences in the batch
        inputs = self.tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        return {
            'input_ids': inputs['input_ids'].astype(np.int32),
            'attention_mask': inputs['attention_mask'].astype(np.int32)
        }
    
    def process_batch(self, sentences: List[str]) -> List[Dict]:
        model_input = self.prepare_batch(sentences)
        
        # Time the prediction
        start_time = time.time()
        predictions = self.model.predict(model_input)
        inference_time = time.time() - start_time
        
        # Process predictions
        output_key = list(predictions.keys())[0]
        logits = predictions[output_key]
        probabilities = 1 / (1 + np.exp(-logits))
        
        results = []
        for i, sentence in enumerate(sentences):
            sentiment = "Positive" if probabilities[i][1] > 0.5 else "Negative"
            confidence = max(probabilities[i][0], probabilities[i][1])
            results.append({
                'text': sentence,
                'sentiment': sentiment,
                'confidence': confidence,
                'inference_time': inference_time / len(sentences)
            })
        
        return results

def test_sentiment_analysis():
    # Initialize analyzer
    analyzer = SentimentAnalyzer(batch_size=64)
    
    # Test sentences (expanded for batch testing)
    test_sentences = [
        "This movie was fantastic! I really enjoyed it.",
        "The service was terrible and the food was cold.",
        "The product works as expected, nothing special.",
        "I can't believe how amazing this experience was!",
        "Very disappointed with the quality.",
        # Add more sentences to test batch processing
        "The new features are incredible and intuitive.",
        "This update completely broke my workflow.",
        "Decent performance but could be better.",
        "Absolutely love the attention to detail!",
        "Not worth the price, save your money."
    ] * 6  # Multiply to create a larger batch
    
    print("\nTesting sentiment analysis with batch processing:")
    print("=" * 50)
    
    # Process in batches
    total_time = 0
    total_samples = 0
    
    for i in range(0, len(test_sentences), analyzer.batch_size):
        batch = test_sentences[i:i + analyzer.batch_size]
        results = analyzer.process_batch(batch)
        
        # Print results and collect metrics
        for result in results:
            print(f"\nInput: {result['text']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Inference time per sample: {result['inference_time']*1000:.2f}ms")
            
            total_time += result['inference_time']
            total_samples += 1
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print("=" * 50)
    print(f"Total samples processed: {total_samples}")
    print(f"Average inference time per sample: {(total_time/total_samples)*1000:.2f}ms")
    print(f"Throughput: {total_samples/total_time:.2f} samples/second")

if __name__ == "__main__":
    test_sentiment_analysis()
