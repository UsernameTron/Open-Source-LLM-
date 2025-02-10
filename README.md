# CoreML LLM Engine

A high-performance LLM execution engine optimized for Apple Silicon, providing real-time sentiment analysis, research insights, and model explainability.

## Features

- Native CoreML execution of transformer models
- Metal Performance Shaders (MPS) GPU acceleration
- Optimized batch processing and quantization
- FastAPI-based API integration
- DuckDB-powered storage and retrieval
- Docker containerization

## System Requirements

- MacOS with Apple Silicon (Optimized for M4 Pro)
- 48GB RAM recommended
- Xcode 15+ with CoreML tools
- Python 3.11+

## Project Structure

```
llm-engine/
├── api/                 # FastAPI application
├── core/               # Core ML engine
│   ├── models/        # CoreML model conversion
│   ├── inference/     # Inference optimization
│   └── processing/    # Text processing
├── storage/           # DuckDB integration
├── benchmarks/        # Performance testing
└── docker/           # Containerization
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

## Usage

1. Start the API server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

2. Run with Docker:
```bash
docker-compose up
```

## Performance Benchmarks

Detailed benchmarks comparing CoreML vs PyTorch CPU execution are available in the `benchmarks/` directory.

## License

MIT License
