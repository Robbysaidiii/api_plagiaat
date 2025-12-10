# ===================================
# README.md
# ===================================

"""
# SkripsiCheck Pro API

API untuk deteksi plagiat berbasis machine learning menggunakan TF-IDF dan Cosine Similarity.

## Features

- 🔍 Deteksi plagiat menggunakan machine learning
- 📄 Support multiple format: PDF, DOC, DOCX, TXT
- 🎯 Real-time analysis dengan similarity scoring
- 📊 Detailed plagiarism reports
- 🚀 RESTful API dengan FastAPI
- 📖 Interactive API documentation

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd skripsicheck-pro-api

# Install requirements
pip install -r requirements.txt

# Run setup script
python run_server.py
```

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Using Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Or manually
docker build -t skripsicheck-api .
docker run -p 8000:8000 skripsicheck-api
```

## API Endpoints

### Health Check
```
GET /
GET /api/health
```

### Plagiarism Detection
```
POST /api/plagiarism-check
Content-Type: multipart/form-data

files: List[UploadFile]
```

Response:
```json
{
  "success": true,
  "judul_dokumen": "Document Title",
  "plagiarism_percentage": 18.5,
  "total_words": 1250,
  "plagiarized_words": 231,
  "references_found": 3,
  "analysis_time": "2.1 detik",
  "uniqueness_score": "4.1 dari 5",
  "original_text": "...",
  "plagiarism_details": [...],
  "similarity_results": [...]
}
```

### Model Status
```
GET /api/models/status
POST /api/models/reload
```

## Frontend Integration

API ini dirancang untuk berintegrasi dengan frontend Quasar.js. Response format sesuai dengan ekspektasi frontend.

## Configuration

Edit `config.py` atau set environment variables:

```bash
export MODEL_PATH="/path/to/your/models"
export MAX_FILE_SIZE=10485760
export SIMILARITY_THRESHOLD=0.3
export PORT=8000
```

## Development

```bash
# Run in development mode
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest tests/

# Check code quality
flake8 .
black .
```

## Deployment

### Production Server

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Model Training

Gunakan Jupyter notebook yang disediakan untuk training model ML:

1. Upload dokumen referensi ke Google Drive
2. Run notebook untuk training
3. Model akan tersimpan di `/content/drive/MyDrive/plagiarism_detection_models/`

## Troubleshooting

### Model Not Found
- Pastikan path model sudah benar
- Run training notebook terlebih dahulu
- Check permissions di Google Drive

### Memory Issues
- Gunakan sparse matrix untuk TF-IDF
- Batasi ukuran dokumen input
- Optimize batch processing

### CORS Issues
- Update CORS_ORIGINS di config
- Check frontend URL configuration

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

MIT License - see LICENSE file for details.
"""