"""Training sequence optimization for fine-tuning."""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)

from .preprocessor import DataPreprocessor, ProcessedExample
from .database import StatisticalDatabase
from .voice import VoiceVerifier, VoiceMetrics

logger = logging.getLogger(__name__)

@dataclass
class TrainingStage:
    """Represents a stage in the training sequence."""
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    grad_accum_steps: int
    loss_weights: Dict[str, float]

@dataclass
class TrainingMetrics:
    """Training metrics for a checkpoint."""
    loss: float
    perplexity: float
    voice_metrics: VoiceMetrics
    technical_accuracy: float
    timestamp: datetime

class StyleLoss(nn.Module):
    """Custom loss function combining content and style metrics."""
    
    def __init__(
        self,
        content_weight: float = 0.7,
        style_weight: float = 0.3
    ):
        """Initialize loss function."""
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        outputs,
        labels,
        style_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate combined loss."""
        content_loss = self.content_loss(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1)
        )
        
        if style_scores is not None:
            style_loss = 1.0 - style_scores.mean()
            return (
                self.content_weight * content_loss +
                self.style_weight * style_loss
            )
        
        return content_loss

class TrainingDataset(Dataset):
    """Dataset for fine-tuning."""
    
    def __init__(
        self,
        examples: List[ProcessedExample],
        tokenizer,
        max_length: int = 1024
    ):
        """Initialize dataset."""
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize text
        inputs = self.tokenizer(
            example.original_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (shift right)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

class ModelTrainer:
    """Manages the fine-tuning process."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device: str = "mps",
        fp16: bool = True
    ):
        """Initialize trainer."""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.fp16 = fp16
        
        # Initialize components
        self._init_components()
        self._setup_training_stages()
        
    def _init_components(self):
        """Initialize all required components."""
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            ).to(self.device)
            
            if self.fp16:
                self.model = self.model.half()
                
            # Initialize support components
            self.preprocessor = DataPreprocessor()
            self.voice_verifier = VoiceVerifier(device=self.device)
            self.stats_db = StatisticalDatabase()
            
            # Initialize loss function
            self.criterion = StyleLoss()
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
            
    def _setup_training_stages(self):
        """Setup training stages sequence."""
        self.stages = [
            TrainingStage(
                name="style_imitation",
                epochs=3,
                batch_size=4,
                learning_rate=2e-5,
                grad_accum_steps=4,
                loss_weights={
                    "content": 0.3,
                    "style": 0.7
                }
            ),
            TrainingStage(
                name="topic_expertise",
                epochs=2,
                batch_size=4,
                learning_rate=1e-5,
                grad_accum_steps=4,
                loss_weights={
                    "content": 0.6,
                    "style": 0.4
                }
            ),
            TrainingStage(
                name="data_integration",
                epochs=2,
                batch_size=4,
                learning_rate=1e-5,
                grad_accum_steps=4,
                loss_weights={
                    "content": 0.7,
                    "style": 0.3
                }
            ),
            TrainingStage(
                name="one_liner_refinement",
                epochs=1,
                batch_size=4,
                learning_rate=5e-6,
                grad_accum_steps=4,
                loss_weights={
                    "content": 0.5,
                    "style": 0.5
                }
            )
        ]
        
    async def train(
        self,
        train_texts: List[str],
        eval_texts: List[str],
        checkpoint_dir: Optional[str] = None
    ):
        """Run complete training sequence."""
        try:
            # Process examples
            train_examples = [
                self.preprocessor.process_example(text)
                for text in train_texts
            ]
            eval_examples = [
                self.preprocessor.process_example(text)
                for text in eval_texts
            ]
            
            # Create datasets
            train_dataset = TrainingDataset(
                train_examples,
                self.tokenizer
            )
            eval_dataset = TrainingDataset(
                eval_examples,
                self.tokenizer
            )
            
            # Train through stages
            best_metrics = None
            for stage in self.stages:
                logger.info(f"Starting {stage.name} stage")
                
                # Update loss weights
                self.criterion = StyleLoss(
                    content_weight=stage.loss_weights["content"],
                    style_weight=stage.loss_weights["style"]
                )
                
                # Train for stage epochs
                stage_metrics = await self._train_stage(
                    stage,
                    train_dataset,
                    eval_dataset
                )
                
                # Save checkpoint if best so far
                if (
                    best_metrics is None or
                    stage_metrics.voice_metrics.overall_score >
                    best_metrics.voice_metrics.overall_score
                ):
                    best_metrics = stage_metrics
                    self._save_checkpoint(
                        stage.name,
                        stage_metrics,
                        checkpoint_dir
                    )
                    
            return best_metrics
            
        except Exception as e:
            logger.error(f"Error in training sequence: {str(e)}")
            raise
            
    async def _train_stage(
        self,
        stage: TrainingStage,
        train_dataset: TrainingDataset,
        eval_dataset: TrainingDataset
    ) -> TrainingMetrics:
        """Train for a single stage."""
        try:
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=stage.batch_size,
                shuffle=True
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=stage.batch_size
            )
            
            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=stage.learning_rate
            )
            
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train_loader) * stage.epochs
            )
            
            # Training loop
            best_eval_loss = float("inf")
            steps_since_best = 0
            
            for epoch in range(stage.epochs):
                # Train epoch
                train_loss = await self._train_epoch(
                    train_loader,
                    optimizer,
                    scheduler,
                    stage.grad_accum_steps
                )
                
                # Evaluate
                eval_metrics = await self._evaluate(
                    eval_loader
                )
                
                logger.info(
                    f"Epoch {epoch + 1}/{stage.epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Eval Loss: {eval_metrics.loss:.4f} - "
                    f"Voice Score: {eval_metrics.voice_metrics.overall_score:.4f}"
                )
                
                # Early stopping check
                if eval_metrics.loss < best_eval_loss:
                    best_eval_loss = eval_metrics.loss
                    steps_since_best = 0
                else:
                    steps_since_best += 1
                    
                if steps_since_best >= 2:  # Early stopping patience
                    logger.info("Early stopping triggered")
                    break
                    
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Error in stage training: {str(e)}")
            raise
            
    async def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        grad_accum_steps: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            try:
                # Move batch to device
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # Calculate loss
                loss = outputs.loss / grad_accum_steps
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update weights if gradient accumulated
                if (i + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
            except Exception as e:
                logger.error(f"Error in training step: {str(e)}")
                continue
                
        return total_loss / len(train_loader)
        
    async def _evaluate(
        self,
        eval_loader: DataLoader
    ) -> TrainingMetrics:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                try:
                    # Move batch to device
                    batch = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                    }
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    total_loss += outputs.loss.item()
                    
                    # Get predictions
                    preds = torch.argmax(
                        outputs.logits,
                        dim=-1
                    )
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(
                        batch["labels"].cpu().numpy()
                    )
                    
                except Exception as e:
                    logger.error(f"Error in evaluation step: {str(e)}")
                    continue
                    
        # Calculate metrics
        avg_loss = total_loss / len(eval_loader)
        perplexity = torch.exp(
            torch.tensor(avg_loss)
        ).item()
        
        # Generate sample text for voice verification
        sample_text = self._generate_sample_text()
        voice_metrics = self.voice_verifier.verify_content(
            sample_text
        )
        
        # Calculate technical accuracy
        valid_preds = [
            p for p, l in zip(all_preds, all_labels)
            if l != -100
        ]
        valid_labels = [
            l for l in all_labels
            if l != -100
        ]
        accuracy = np.mean(
            np.array(valid_preds) == np.array(valid_labels)
        )
        
        return TrainingMetrics(
            loss=avg_loss,
            perplexity=perplexity,
            voice_metrics=voice_metrics,
            technical_accuracy=accuracy,
            timestamp=datetime.now()
        )
        
    def _generate_sample_text(
        self,
        prompt: str = "Let's discuss",
        max_length: int = 200
    ) -> str:
        """Generate sample text for evaluation."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            return self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
        except Exception as e:
            logger.error(f"Error generating sample: {str(e)}")
            return ""
            
    def _save_checkpoint(
        self,
        stage_name: str,
        metrics: TrainingMetrics,
        checkpoint_dir: Optional[str] = None
    ):
        """Save model checkpoint."""
        try:
            # Create checkpoint directory
            save_dir = Path(checkpoint_dir or self.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            checkpoint_path = save_dir / f"checkpoint_{stage_name}"
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            
            # Save metrics
            metrics_dict = {
                "loss": metrics.loss,
                "perplexity": metrics.perplexity,
                "voice_metrics": {
                    "rhythm_score": metrics.voice_metrics.rhythm_score,
                    "humor_score": metrics.voice_metrics.humor_score,
                    "skepticism_score": metrics.voice_metrics.skepticism_score,
                    "data_usage_score": metrics.voice_metrics.data_usage_score,
                    "one_liner_score": metrics.voice_metrics.one_liner_score,
                    "overall_score": metrics.voice_metrics.overall_score
                },
                "technical_accuracy": metrics.technical_accuracy,
                "timestamp": metrics.timestamp.isoformat()
            }
            
            metrics_path = checkpoint_path / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        try:
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path
            )
            
            if self.fp16:
                self.model = self.model.half()
                
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
