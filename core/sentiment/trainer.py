import logging
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report
from datasets import load_dataset
import nlpaug.augmenter.word as naw
import os
import json
import torch
import shap
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup, DataCollatorWithPadding
from rich.console import Console
import sys
import traceback
import plotly.graph_objects as go

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentTrainer:
    """Trainer class for sentiment analysis."""
    
    def __init__(self, hf_token=None):
        """Initialize the sentiment trainer"""
        self.hf_token = hf_token
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = 16
        self.epochs = 3
        self.learning_rate = 2e-5
        self.warmup_steps = 500
        self.status_file = Path("training_status.json")
        
        print("\nInitializing trainer with device:", self.device, flush=True)
        
        print("\nLoading model and tokenizer...", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large",
            num_labels=3,
            use_auth_token=self.hf_token
        ).to(self.device)
        print("✓ Model loaded successfully", flush=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-large",
            use_auth_token=self.hf_token
        )
        print("✓ Tokenizer loaded successfully", flush=True)
        
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        print("✓ Data collator initialized", flush=True)
        
        print("\nTraining parameters:", flush=True)
        print(f"- Batch size: {self.batch_size}", flush=True)
        print(f"- Epochs: {self.epochs}", flush=True)
        print(f"- Learning rate: {self.learning_rate}", flush=True)
        print(f"- Warmup steps: {self.warmup_steps}", flush=True)

    def update_status(self, status, **kwargs):
        """Update training status"""
        try:
            status_data = {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                **kwargs
            }
            
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update status file: {e}", flush=True)

    def load_datasets(self):
        """Load all datasets"""
        print("\n=== Starting Dataset Loading ===\n", flush=True)
        
        all_texts = []
        all_labels = []
        loaded_datasets = []
        
        # Twitter dataset (more reliable)
        try:
            print("Loading Twitter sentiment dataset...", flush=True)
            dataset = load_dataset(
                "cardiffnlp/tweet_sentiment_multilingual",
                "english",
                token=self.hf_token,
                trust_remote_code=True
            )
            texts = dataset["train"]["text"]
            # Convert labels: NEG=0, NEU=1, POS=2
            labels = dataset["train"]["label"]
            all_texts.extend(texts)
            all_labels.extend(labels)
            loaded_datasets.append(("twitter", len(texts)))
            print(f"✓ Loaded {len(texts)} samples from Twitter dataset", flush=True)
        except Exception as e:
            print(f"Warning: Error loading Twitter dataset: {e}", flush=True)

        # Yelp Reviews (replacement for Amazon)
        try:
            print("\nLoading Yelp reviews dataset...", flush=True)
            dataset = load_dataset(
                "yelp_review_full",
                token=self.hf_token,
                trust_remote_code=True
            )
            texts = dataset["train"]["text"]
            # Convert 1-5 star ratings to sentiment: 1-2=NEG, 3=NEU, 4-5=POS
            labels = [0 if r <= 2 else 2 if r >= 4 else 1 for r in dataset["train"]["stars"]]
            texts = texts[:10000]  # Limit size
            labels = labels[:10000]
            all_texts.extend(texts)
            all_labels.extend(labels)
            loaded_datasets.append(("yelp", len(texts)))
            print(f"✓ Loaded {len(texts)} samples from Yelp dataset", flush=True)
        except Exception as e:
            print(f"Warning: Error loading Yelp dataset: {e}", flush=True)
            
        # IMDB Reviews (reliable fallback)
        if not all_texts:
            try:
                print("\nLoading IMDB reviews dataset...", flush=True)
                dataset = load_dataset(
                    "imdb",
                    token=self.hf_token,
                    trust_remote_code=True
                )
                texts = dataset["train"]["text"]
                # Convert binary labels to our format (0=neg, 2=pos)
                labels = [0 if label == 0 else 2 for label in dataset["train"]["label"]]
                texts = texts[:10000]  # Limit size
                labels = labels[:10000]
                all_texts.extend(texts)
                all_labels.extend(labels)
                loaded_datasets.append(("imdb", len(texts)))
                print(f"✓ Loaded {len(texts)} samples from IMDB dataset", flush=True)
            except Exception as e:
                print(f"Warning: Error loading IMDB dataset: {e}", flush=True)

        if not all_texts:
            raise ValueError("Failed to load any datasets. Cannot proceed with training.")

        print(f"\n=== Dataset Loading Summary ===")
        print(f"Total samples: {len(all_texts)}")
        for name, count in loaded_datasets:
            print(f"- {name}: {count} samples")
        
        return all_texts, all_labels

    def balance_dataset(self, texts, labels):
        """Balance dataset"""
        label_counts = np.bincount(labels)
        max_count = max(label_counts)
        logger.info(f"Initial label distribution: {label_counts}")
        
        balanced_texts = []
        balanced_labels = []
        
        for label in range(3):
            label_texts = [t for t, l in zip(texts, labels) if l == label]
            logger.info(f"Processing label {label} with {len(label_texts)} samples")
            
            # Original samples
            balanced_texts.extend(label_texts)
            balanced_labels.extend([label] * len(label_texts))
            
            # Augment if needed
            if len(label_texts) < max_count:
                num_augment = max_count - len(label_texts)
                logger.info(f"Augmenting label {label} with {num_augment} samples")
                for _ in range(num_augment):
                    text = random.choice(label_texts)
                    aug_text = naw.SynonymAug(aug_src='wordnet', aug_p=0.3).augment(text)[0]
                    balanced_texts.append(aug_text)
                    balanced_labels.append(label)
        
        # Shuffle
        combined = list(zip(balanced_texts, balanced_labels))
        random.shuffle(combined)
        balanced_texts, balanced_labels = zip(*combined)
        
        return balanced_texts, balanced_labels

    def create_dataset(self, texts, labels):
        """Create dataset"""
        dataset = SentimentDataset(texts, labels, self.tokenizer)
        return dataset

    def explain_prediction(self, text):
        """Generate SHAP explanation for a prediction"""
        self.model.eval()
        explainer = shap.Explainer(
            lambda x: self.model(
                self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
            ).logits.cpu().detach().numpy(),
            self.tokenizer
        )
        shap_values = explainer([text])
        
        # Convert to visualization
        fig = go.Figure()
        words = self.tokenizer.tokenize(text)
        for i, word in enumerate(words):
            fig.add_trace(go.Bar(
                x=[shap_values.values[0][i]],
                y=[word],
                orientation='h'
            ))
        
        fig.update_layout(
            title="Word Importance for Sentiment",
            xaxis_title="SHAP value (impact on prediction)",
            yaxis_title="Token"
        )
        
        return {
            "shap_values": shap_values.values.tolist(),
            "tokens": words,
            "plot": fig.to_json()
        }

    def train(self, train_loader):
        """Train the model."""
        print("\nStarting training...", flush=True)
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            print(f"\nEpoch {epoch + 1}/{self.epochs}:", flush=True)
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}", flush=True)
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print epoch summary
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch + 1} summary:", flush=True)
            print(f"- Average loss: {avg_loss:.4f}", flush=True)
        
        print("\nTraining completed successfully!", flush=True)

    def evaluate(self, train_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in train_loader:
                outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = classification_report(
            all_labels,
            all_preds,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        return avg_loss, metrics
    
    def save_model(self):
        """Save model and tokenizer"""
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

def main():
    """Main training function"""
    status_file = Path("training_status.json")
    
    try:
        print("\n=== Starting Sentiment Analysis Training ===", flush=True)
        
        # Get Hugging Face token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("Error: HUGGINGFACE_TOKEN environment variable not set", flush=True)
            sys.exit(1)
        
        # Clear MPS cache and optimize memory
        if torch.backends.mps.is_available():
            print("\nOptimizing MPS device...", flush=True)
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.8)
        
        # Initialize trainer with progress tracking
        print("\nStep 1: Initializing trainer...", flush=True)
        trainer = SentimentTrainer(hf_token=hf_token)
        
        # Load datasets with progress
        print("\nStep 2: Loading datasets...", flush=True)
        texts, labels = trainer.load_datasets()
        print(f"✓ Loaded initial datasets: {len(texts)} samples", flush=True)
        
        # Balance dataset
        print("\nStep 3: Balancing datasets...", flush=True)
        balanced_texts, balanced_labels = trainer.balance_dataset(texts, labels)
        print(f"✓ Balanced dataset: {len(balanced_texts)} samples", flush=True)
        
        # Create dataset and loader
        print("\nStep 4: Creating data loader...", flush=True)
        dataset = trainer.create_dataset(balanced_texts, balanced_labels)
        train_loader = DataLoader(
            dataset,
            batch_size=trainer.batch_size,
            shuffle=True,
            collate_fn=trainer.data_collator
        )
        print(f"✓ Created data loader with {len(train_loader)} batches", flush=True)
        
        # Start training with enhanced monitoring
        print("\nStep 5: Starting training...", flush=True)
        print("\nTraining Configuration:", flush=True)
        print(f"- Device: {trainer.device}", flush=True)
        print(f"- Batch Size: {trainer.batch_size}", flush=True)
        print(f"- Learning Rate: {trainer.learning_rate}", flush=True)
        print(f"- Epochs: {trainer.epochs}", flush=True)
        print(f"- Total Batches: {len(train_loader)}", flush=True)
        print(f"- Samples per Epoch: {len(dataset)}", flush=True)
        
        # Start training
        trainer.train(train_loader)
        
        print("\n=== Training Complete ===", flush=True)
        trainer.update_status("completed")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}", flush=True)
        traceback.print_exc()
        try:
            trainer.update_status("error", error=str(e))
        except:
            with open(status_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e)
                }, f, indent=2)
        sys.exit(1)

if __name__ == '__main__':
    main()
