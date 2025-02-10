import io
import csv
import logging
import mimetypes
from typing import Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from middleware import rate_limit_middleware, verify_api_key_middleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Ensure PDF mimetype is registered
mimetypes.add_type('application/pdf', '.pdf')

class TextRequest(BaseModel):
    text: str
import random
import time

app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add middleware
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(verify_api_key_middleware)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add health check endpoint
@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class TextRequest(BaseModel):
    text: str

def get_ngrams(text: str, n: int) -> list:
    """Generate n-grams from text."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def analyze_sentiment(text: str):
    # Enhanced phrase dictionaries
    contextual_phrases = {
        # Strong positive phrases (weight 2.5)
        'exceeded expectations': 2.5, 'significant improvement': 2.5,
        'major breakthrough': 2.5, 'outstanding performance': 2.5,
        'highly effective': 2.5, 'exceptional quality': 2.5,
        'perfect solution': 2.5, 'remarkable achievement': 2.5,
        
        # Strong negative phrases (weight -2.5)
        'completely failed': -2.5, 'serious problem': -2.5,
        'major issue': -2.5, 'critical failure': -2.5,
        'severely impacted': -2.5, 'extremely poor': -2.5,
        'totally unusable': -2.5, 'absolutely terrible': -2.5,
        
        # Technical positive phrases (weight 2.0)
        'improved performance': 2.0, 'enhanced stability': 2.0,
        'better scalability': 2.0, 'optimized throughput': 2.0,
        'reduced latency': 2.0, 'increased reliability': 2.0,
        
        # Technical negative phrases (weight -2.0)
        'degraded performance': -2.0, 'increased latency': -2.0,
        'reduced throughput': -2.0, 'system crash': -2.0,
        'memory leak': -2.0, 'data corruption': -2.0
    }
    
    # Negation phrases that flip sentiment
    negation_phrases = {
        'not': -1, 'no': -1, 'never': -1, 'none': -1,
        'hardly': -0.5, 'barely': -0.5, 'scarcely': -0.5,
        'doesn\'t': -1, 'don\'t': -1, 'didn\'t': -1,
        'wouldn\'t': -1, 'couldn\'t': -1, 'shouldn\'t': -1,
        'won\'t': -1, 'can\'t': -1, 'isn\'t': -1
    }
    
    # Sarcasm indicators with weights
    sarcasm_indicators = {
        'oh great': 0.8, 'wow': 0.6, 'amazing': 0.7,
        'sure': 0.6, 'clear as': 0.9, 'another': 0.5,
        'supposedly': 0.8, 'air quotes': 0.9,
        'just what we needed': 0.8, 'brilliant': 0.7
    }
    
    text_lower = text.lower()
    
    # Check for sarcasm using n-grams
    sarcasm_score = 0
    bigrams = get_ngrams(text_lower, 2)
    trigrams = get_ngrams(text_lower, 3)
    
    for indicator, weight in sarcasm_indicators.items():
        if indicator in text_lower:
            sarcasm_score += weight
    
    sarcasm_detected = sarcasm_score > 0.7  # Threshold for sarcasm detection
    """Enhanced sentiment analysis with context awareness and weighted scoring."""
    text = text.lower()
    
    # Expanded word lists with weights
    positive_words = {
        # Strong positive (weight 2)
        'exceptional': 2, 'outstanding': 2, 'breakthrough': 2, 'remarkable': 2,
        'excellent': 2, 'innovative': 2, 'revolutionary': 2, 'transformative': 2,
        'extraordinary': 2, 'tremendous': 2, 'exceeded': 2,
        
        # Regular positive (weight 1)
        'good': 1, 'great': 1, 'successful': 1, 'improved': 1, 'effective': 1,
        'efficient': 1, 'positive': 1, 'valuable': 1, 'beneficial': 1, 'better': 1,
        'progress': 1, 'achievement': 1, 'solved': 1, 'solution': 1, 'love': 1,
        
        # Technical positive (weight 1.5)
        'robust': 1.5, 'scalable': 1.5, 'reliable': 1.5, 'maintainable': 1.5,
        'optimized': 1.5, 'stable': 1.5, 'consistent': 1.5, 'resilient': 1.5,
        'streamlined': 1.5, 'performant': 1.5
    }
    
    negative_words = {
        # Strong negative (weight 2)
        'critical': 2, 'severe': 2, 'terrible': 2, 'horrible': 2, 'catastrophic': 2,
        'devastating': 2, 'fatal': 2, 'worst': 2, 'failed': 2, 'unusable': 2,
        
        # Regular negative (weight 1)
        'bad': 1, 'poor': 1, 'issue': 1, 'problem': 1, 'bug': 1, 'error': 1,
        'defect': 1, 'flaw': 1, 'concern': 1, 'difficult': 1, 'challenging': 1,
        
        # Technical negative (weight 1.5)
        'unstable': 1.5, 'inconsistent': 1.5, 'unreliable': 1.5, 'bottleneck': 1.5,
        'vulnerability': 1.5, 'deprecated': 1.5, 'legacy': 1.5, 'technical debt': 1.5
    }
    
    # Context modifiers
    improvement_indicators = {'despite', 'however', 'but', 'although', 'resolved', 'improved',
                            'solved', 'overcame', 'achieved', 'succeeded'}
    
    # Split into sentences and generate n-grams
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    total_score = 0
    found_phrases = set()
    improvement_context = False
    
    # Check for contextual phrases
    for n in [2, 3]:
        ngrams = get_ngrams(text_lower, n)
        for phrase in ngrams:
            if phrase in contextual_phrases:
                score = contextual_phrases[phrase]
                found_phrases.add(phrase)
                total_score += score
    
    for sentence in sentences:
        words = sentence.split()
        sentence_score = 0
        negation_multiplier = 1
        
        # Process words in the sentence
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check for negation
            if word_lower in negation_phrases:
                negation_multiplier = negation_phrases[word_lower]
                continue
            
            # Check for improvement context
            if word_lower in improvement_indicators:
                improvement_context = True
            
            # Calculate word sentiment
            if word_lower in positive_words:
                score = positive_words[word_lower]
                if sarcasm_detected:
                    score *= -0.5  # Reduce positive score for sarcasm
                score *= negation_multiplier  # Apply negation
                found_phrases.add(word)
                sentence_score += score
                negation_multiplier = 1  # Reset after applying
            
            elif word_lower in negative_words:
                score = -negative_words[word_lower]
                score *= negation_multiplier  # Apply negation
                found_phrases.add(word)
                if improvement_context:
                    score = abs(score) * 1.5  # Stronger positive for overcome negatives
                sentence_score += score
                negation_multiplier = 1  # Reset after applying
        
        # Weight later sentences more heavily (temporal progression)
        position_weight = 1.0 + (sentences.index(sentence) / len(sentences))
        total_score += sentence_score * position_weight
    
    # Calculate final sentiment with sarcasm adjustment
    if total_score > 0:
        if sarcasm_detected:
            # Convert positive to negative for sarcastic text
            prediction = 'negative'
            confidence = min(0.85 + (len(found_words) * 0.03), 1.0)
        else:
            prediction = 'positive'
            confidence = min(0.8 + (len(found_words) * 0.05), 1.0)
    elif total_score < 0:
        prediction = 'negative'
        # Higher confidence for negative with sarcasm
        confidence = min(0.8 + (len(found_words) * 0.05) + (0.1 if sarcasm_detected else 0), 1.0)
    else:
        prediction = 'neutral'
        confidence = 0.8
    
    # Generate detailed explanation
    explanation = f'Analysis found {len(found_phrases)} sentiment-indicating phrases and words.'
    
    # Add key phrases
    if found_phrases:
        key_phrases = sorted(found_phrases)
        if len(key_phrases) > 5:
            key_phrases = key_phrases[:5] + ['...']
        explanation += f' Key phrases: {", ".join(key_phrases)}'
    
    # Add context information
    if improvement_context:
        explanation += ' Context indicates improvement or resolution of challenges.'
    
    # Add sarcasm detection details
    if sarcasm_detected:
        explanation += f' Sarcasm detected (confidence: {min(sarcasm_score/2, 1.0):.2%}), sentiment adjusted accordingly.'
    
    # Add negation information
    negation_count = sum(1 for word in text.lower().split() if word in negation_phrases)
    if negation_count > 0:
        explanation += f' Found {negation_count} negation phrases that modified the sentiment.'
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'explanation': explanation
    }

@app.post("/api/analyze-text")
async def analyze_text(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    start_time = time.time()
    result = analyze_sentiment(request.text)
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "explanation": result["explanation"],
        "processing_time_ms": round(processing_time, 2)
    }

# ---- Begin Document Upload Endpoint and Helpers ----

# Import PyMuPDF
try:
    import fitz
    logger.info('PyMuPDF imported successfully')
    PDF_SUPPORT = True
except ImportError as e:
    logger.error(f'Failed to import PyMuPDF: {str(e)}')
    PDF_SUPPORT = False


def extract_text_from_pdf(file_obj):
    try:
        file_obj.seek(0)
        pdf_bytes = file_obj.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")


def extract_text_from_csv(file_obj):
    file_obj.seek(0)
    reader = csv.DictReader(io.StringIO(file_obj.read().decode("utf-8")))
    texts = []
    for row in reader:
        if "feedback" in row:
            texts.append(row["feedback"])
        else:
            texts.append(" ".join(row.values()))
    return " ".join(texts)


def extract_text_from_txt(file_obj):
    file_obj.seek(0)
    return file_obj.read().decode("utf-8")


from fastapi import UploadFile, File, HTTPException


@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        # Log incoming file details
        logger.info(f'Processing file: {file.filename}, content_type: {file.content_type}')
        
        # Enforce file size limit of 100MB
        contents = await file.read()
        file_size = len(contents)
        logger.info(f'File size: {file_size / 1024 / 1024:.2f} MB')
        
        if file_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File size exceeds limit of 100MB")
        
        # Create a BytesIO stream from contents
        file_stream = io.BytesIO(contents)
        file_text = ""
        
        # Determine content type
        content_type = file.content_type
        if not content_type:
            # Try to guess content type from filename
            content_type, _ = mimetypes.guess_type(file.filename)
            logger.info(f'Guessed content type: {content_type}')
        
        if not content_type:
            content_type = "text/plain"  # Default to text/plain if still not determined
        
        logger.info(f'Processing file as {content_type}')
        
        if content_type == "application/pdf":
            if not PDF_SUPPORT:
                logger.error('PDF support not available')
                raise HTTPException(status_code=415, detail="PDF support not available. Only TXT and CSV files are supported.")
            logger.info('Extracting text from PDF')
            file_text = extract_text_from_pdf(file_stream)
        elif content_type == "text/csv":
            logger.info('Extracting text from CSV')
            file_text = extract_text_from_csv(file_stream)
        elif content_type == "text/plain":
            logger.info('Extracting text from TXT')
            file_text = extract_text_from_txt(file_stream)
        else:
            logger.error(f'Unsupported content type: {content_type}')
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {content_type}. Allowed types: PDF, CSV, TXT")
        
        if not file_text.strip():
            logger.error('No text content found in file')
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        logger.info(f'Extracted text length: {len(file_text)} characters')
        
        # Re-use the existing analyze_text function for text analysis
        analysis = await analyze_text(TextRequest(text=file_text))
        logger.info('Analysis completed successfully')
        return analysis
    except Exception as e:
        logger.error(f'Error processing file: {str(e)}')
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
# ---- End Document Upload Endpoint and Helpers ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
