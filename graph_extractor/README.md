# Graph Extractor

A production-ready Python tool for extracting graph images and their associated captions from PDF documents.

## Features

- Extracts graph images from PDFs (digital and scanned)
- Associates correct captions using spatial analysis and LLM reasoning
- Supports multiple OCR engines (Tesseract, PaddleOCR)
- Uses Ollama for intelligent graph validation and caption selection
- Outputs structured metadata and optional CSV summary
- Handles batch processing with error recovery

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install system dependencies:

For pdf2image (Poppler):
- Windows: Download from https://blog.alivate.com.au/poppler-windows/
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

For PaddleOCR (optional):
- Install paddlepaddle and paddleocr if needed
- Note: May have compatibility issues on some systems

3. Install Ollama and pull a model:
```bash
ollama pull mistral
```

## Usage

```bash
python main.py input.pdf output_directory/
```

### Options

- `--config CONFIG`: Path to configuration file (default: config.yaml)
- `--verbose, -v`: Enable verbose logging

## Configuration

Edit `config.yaml` to customize:

- PDF processing settings (DPI, Poppler path)
- Layout detection model and thresholds
- Graph detection parameters
- OCR engine selection
- Ollama model settings
- Output options

## Output Structure

```
output_directory/
├── graphs/
│   ├── graph_001.png
│   ├── graph_002.png
├── metadata.json
├── summary.csv (optional)
```

### metadata.json Format

```json
{
  "graph_001.png": {
    "page": 12,
    "caption": "Velocity vs Time graph",
    "type": "line graph",
    "confidence": 0.92
  }
}
```

## Project Structure

```
graph_extractor/
├── main.py                 # CLI entry point
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── extractor/
│   ├── pdf_loader.py      # PDF to image conversion
│   ├── layout_detector.py # Layout block detection
│   ├── graph_detector.py  # Graph region identification
│   └── caption_matcher.py # OCR and caption matching
├── llm/
│   └── ollama_reasoner.py # LLM reasoning for validation
└── utils/
    ├── image_processing.py # Image utilities
    └── logging_helper.py  # Logging setup
```

## Requirements

- Python 3.8+
- Ollama running locally
- Tesseract or PaddleOCR installed
- Poppler for PDF processing

## License

MIT License