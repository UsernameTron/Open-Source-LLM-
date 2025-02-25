"""Voice consistency verification system."""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM
)

logger = logging.getLogger(__name__)

@dataclass
class VoiceMetrics:
    """Metrics for voice consistency."""
    rhythm_score: float
    humor_score: float
    skepticism_score: float
    data_usage_score: float
    one_liner_score: float
    overall_score: float
    details: Dict[str, Any]

class VoiceVerifier:
    """Verifies content adherence to voice style."""
    
    def __init__(
        self,
        device: str = "mps"  # Use Metal Performance Shaders
    ):
        """Initialize verifier models."""
        self.device = device
        self._init_models()
        
    def _init_models(self):
        """Initialize all required models."""
        try:
            # Style classifier
            self.style_tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased"
            )
            self.style_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=5  # Different style aspects
                ).to(self.device)
            )
            
            # Language model for perplexity
            self.lm_tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased"
            )
            self.lm_model = (
                AutoModelForMaskedLM.from_pretrained(
                    "bert-base-uncased"
                ).to(self.device)
            )
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
            
    def verify_content(self, text: str) -> VoiceMetrics:
        """Verify content adherence to voice style."""
        try:
            # Get base metrics
            rhythm_score = self._analyze_rhythm(text)
            humor_score = self._analyze_humor(text)
            skepticism_score = self._analyze_skepticism(text)
            data_score = self._analyze_data_usage(text)
            one_liner_score = self._analyze_one_liner(text)
            
            # Calculate overall score
            overall_score = np.mean([
                rhythm_score,
                humor_score,
                skepticism_score,
                data_score,
                one_liner_score
            ])
            
            return VoiceMetrics(
                rhythm_score=rhythm_score,
                humor_score=humor_score,
                skepticism_score=skepticism_score,
                data_usage_score=data_score,
                one_liner_score=one_liner_score,
                overall_score=overall_score,
                details=self._get_detailed_analysis(text)
            )
            
        except Exception as e:
            logger.error(f"Error verifying content: {str(e)}")
            raise
            
    def _analyze_rhythm(self, text: str) -> float:
        """Analyze rhythm patterns."""
        try:
            # Tokenize sentences
            sentences = text.split(".")
            
            # Calculate sentence length variance
            lengths = [len(s.split()) for s in sentences if s.strip()]
            length_var = np.var(lengths) if lengths else 0
            
            # Calculate rhythm score based on variance
            # (Some variance is good, too much is bad)
            optimal_var = 25  # Empirically determined
            rhythm_score = 1.0 - min(
                abs(length_var - optimal_var) / optimal_var,
                1.0
            )
            
            return float(rhythm_score)
            
        except Exception as e:
            logger.error(f"Error analyzing rhythm: {str(e)}")
            return 0.0
            
    def _analyze_humor(self, text: str) -> float:
        """Analyze humor style consistency."""
        try:
            # Encode text
            inputs = self.style_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.style_model(**inputs)
                
            # Convert to humor score (0-1)
            scores = torch.nn.functional.softmax(
                outputs.logits,
                dim=-1
            )
            humor_score = float(scores[0][0].cpu().numpy())
            
            return humor_score
            
        except Exception as e:
            logger.error(f"Error analyzing humor: {str(e)}")
            return 0.0
            
    def _analyze_skepticism(self, text: str) -> float:
        """Analyze appropriate skepticism level."""
        try:
            # Look for skepticism markers
            markers = [
                "but",
                "however",
                "actually",
                "reality",
                "truth",
                "claim",
                "supposedly"
            ]
            
            # Count markers
            marker_count = sum(
                text.lower().count(m)
                for m in markers
            )
            
            # Normalize score
            words = len(text.split())
            optimal_ratio = 0.05  # 5% skepticism markers
            actual_ratio = marker_count / words
            
            return float(
                1.0 - min(
                    abs(actual_ratio - optimal_ratio) / optimal_ratio,
                    1.0
                )
            )
            
        except Exception as e:
            logger.error(f"Error analyzing skepticism: {str(e)}")
            return 0.0
            
    def _analyze_data_usage(self, text: str) -> float:
        """Analyze data-backed assertions."""
        try:
            # Look for numerical patterns
            number_patterns = [
                r"\d+%",
                r"\$\d+",
                r"\d+x",
                r"\d+\s+times",
                r"\d+\s+percent"
            ]
            
            # Count patterns
            pattern_count = sum(
                len(re.findall(pattern, text))
                for pattern in number_patterns
            )
            
            # Look for citation markers
            citation_markers = [
                "according to",
                "reports",
                "studies show",
                "research indicates",
                "data suggests"
            ]
            
            citation_count = sum(
                text.lower().count(marker)
                for marker in citation_markers
            )
            
            # Calculate combined score
            paragraphs = text.count("\n\n") + 1
            optimal_counts = {
                "numbers": paragraphs * 2,  # 2 numbers per paragraph
                "citations": paragraphs  # 1 citation per paragraph
            }
            
            number_score = min(
                pattern_count / optimal_counts["numbers"],
                1.0
            )
            citation_score = min(
                citation_count / optimal_counts["citations"],
                1.0
            )
            
            return float((number_score + citation_score) / 2)
            
        except Exception as e:
            logger.error(f"Error analyzing data usage: {str(e)}")
            return 0.0
            
    def _analyze_one_liner(self, text: str) -> float:
        """Analyze one-liner quality."""
        try:
            # Find potential one-liner
            paragraphs = text.split("\n\n")
            last_para = paragraphs[-1].strip()
            
            if not last_para.startswith("**One-Liner"):
                return 0.0
                
            # Analyze one-liner
            one_liner = last_para.replace("**One-Liner**:", "").strip()
            
            # Calculate metrics
            metrics = {
                "length": 1.0 - abs(len(one_liner.split()) - 15) / 15,
                "memorability": self._calculate_memorability(one_liner),
                "relevance": self._calculate_relevance(
                    one_liner,
                    "\n".join(paragraphs[:-1])
                )
            }
            
            return float(np.mean(list(metrics.values())))
            
        except Exception as e:
            logger.error(f"Error analyzing one-liner: {str(e)}")
            return 0.0
            
    def _calculate_memorability(self, text: str) -> float:
        """Calculate text memorability score."""
        try:
            # Encode text
            inputs = self.lm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = self.lm_model(**inputs)
                
            # Lower perplexity = more memorable
            perplexity = torch.exp(
                outputs.loss
            ).cpu().numpy()
            
            # Convert to score (0-1)
            return float(
                1.0 - min(perplexity / 100, 1.0)
            )
            
        except Exception as e:
            logger.error(
                f"Error calculating memorability: {str(e)}"
            )
            return 0.0
            
    def _calculate_relevance(
        self,
        one_liner: str,
        context: str
    ) -> float:
        """Calculate one-liner relevance to context."""
        try:
            # Encode texts
            inputs = self.style_tokenizer(
                context,
                one_liner,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.style_model(**inputs)
                
            # Convert to relevance score
            scores = torch.nn.functional.softmax(
                outputs.logits,
                dim=-1
            )
            relevance_score = float(
                scores[0][1].cpu().numpy()
            )
            
            return relevance_score
            
        except Exception as e:
            logger.error(
                f"Error calculating relevance: {str(e)}"
            )
            return 0.0
            
    def _get_detailed_analysis(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Get detailed analysis of voice characteristics."""
        return {
            "sentence_structure": self._analyze_sentence_structure(text),
            "vocabulary_usage": self._analyze_vocabulary(text),
            "rhetorical_devices": self._analyze_rhetorical_devices(text),
            "tone_consistency": self._analyze_tone(text)
        }
        
    def _analyze_sentence_structure(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Analyze sentence structure patterns."""
        sentences = text.split(".")
        return {
            "avg_length": np.mean([
                len(s.split())
                for s in sentences if s.strip()
            ]),
            "length_variance": np.var([
                len(s.split())
                for s in sentences if s.strip()
            ]),
            "complex_sentences": sum(
                1 for s in sentences
                if len(s.split()) > 20
            ) / len(sentences)
        }
        
    def _analyze_vocabulary(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Analyze vocabulary usage."""
        words = text.lower().split()
        return {
            "unique_ratio": len(set(words)) / len(words),
            "tech_terms": self._count_tech_terms(text),
            "business_jargon": self._count_jargon(text)
        }
        
    def _analyze_rhetorical_devices(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Analyze use of rhetorical devices."""
        return {
            "questions": text.count("?"),
            "analogies": self._count_analogies(text),
            "contrasts": self._count_contrasts(text)
        }
        
    def _analyze_tone(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Analyze tone consistency."""
        return {
            "formality": self._measure_formality(text),
            "sentiment_variance": self._measure_sentiment_variance(text),
            "engagement_level": self._measure_engagement(text)
        }
