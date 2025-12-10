from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import io
import os
import tempfile
import json
import numpy as np
from datetime import datetime
import uuid
import logging
import hashlib

# Machine Learning Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle

# Report Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Global variables
detector = None
DOCUMENTS_DIR = "documents"
MODELS_DIR = "models"
RESULTS_DIR = "results"

class PlagiarismDetector:
    def __init__(self):
        self.vectorizer = None
        self.reference_docs = []
        self.reference_vectors = None
        self.reference_metadata = []
        self.stemmer = PorterStemmer()
        
        # Setup stopwords
        try:
            self.stop_words = set(stopwords.words('indonesian'))
        except:
            self.stop_words = set(stopwords.words('english'))
            logger.info("✅ Using English stopwords")
        
        # Load atau train model
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load model yang sudah ada atau train baru"""
        model_path = os.path.join(MODELS_DIR, "plagiarism_model.pkl")
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.reference_docs = model_data['reference_docs']
                    self.reference_vectors = model_data['reference_vectors']
                    self.reference_metadata = model_data.get('reference_metadata', [])
                logger.info(f"✅ Model loaded successfully with {len(self.reference_docs)} references")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                self.initialize_new_model()
        else:
            self.initialize_new_model()
    
    def initialize_new_model(self):
        """Initialize model baru"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3),
            stop_words=list(self.stop_words)
        )
        logger.info("✅ New model initialized")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess teks: lowercase, tokenize, remove stopwords, stemming"""
        try:
            # Lowercase
            text = text.lower()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords dan punctuation
            tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
            
            # Stemming
            tokens = [self.stemmer.stem(word) for word in tokens]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {e}")
            return text.lower()  # Fallback
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract teks dari file PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.info(f"📄 PDF has {num_pages} pages")
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    if i < 3:  # Log first 3 pages
                        logger.info(f"📝 Page {i+1}: {len(page_text or '')} chars")
                
            text = text.strip()
            logger.info(f"✅ Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"❌ PDF extraction error: {e}")
            raise HTTPException(status_code=400, detail=f"Gagal membaca file PDF: {str(e)}")
    
    def add_reference_document(self, file_path: str, filename: str, metadata: dict = None):
        """Tambah dokumen referensi ke model"""
        try:
            logger.info(f"📥 Adding reference: {filename}")
            
            # Extract text
            text = self.extract_text_from_pdf(file_path)
            if not text or len(text) < 50:
                raise ValueError("Dokumen terlalu pendek atau tidak mengandung teks")
            
            # Preprocess
            processed_text = self.preprocess_text(text)
            logger.info(f"✅ Preprocessed text: {len(processed_text)} characters")
            
            # Simpan dokumen
            self.reference_docs.append(processed_text)
            
            # Simpan metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                'filename': filename,
                'original_text': text[:1000],  # Simpan sebagian teks asli
                'processed_length': len(processed_text),
                'added_date': datetime.now().isoformat()
            })
            self.reference_metadata.append(doc_metadata)
            
            # Update TF-IDF vectors
            if self.vectorizer:
                if self.reference_vectors is None:
                    # Fit dan transform untuk pertama kali
                    self.reference_vectors = self.vectorizer.fit_transform([processed_text])
                else:
                    # Fit ulang dengan dokumen baru
                    self.reference_vectors = self.vectorizer.fit_transform(self.reference_docs)
            
            # Save model
            self.save_model()
            
            logger.info(f"✅ Referensi ditambahkan: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Gagal menambahkan referensi: {e}")
            return False
    
    def check_plagiarism(self, text: str, filename: str) -> Dict:
        """Cek plagiarisme pada teks input"""
        try:
            start_time = datetime.now()
            logger.info(f"🔍 Starting plagiarism check for: {filename}")
            logger.info(f"📝 Input text length: {len(text)} characters")
            
            # Preprocess teks input
            processed_text = self.preprocess_text(text)
            logger.info(f"✅ Preprocessed length: {len(processed_text)} characters")
            
            if not processed_text or len(processed_text.split()) < 10:
                logger.warning("⚠️ Text terlalu pendek untuk dianalisis")
                return {
                    'success': True,
                    'message': 'Teks terlalu pendek untuk dianalisis',
                    'judul_skripsi': filename.replace('.pdf', ''),
                    'tingkat_kesamaan': 0,
                    'kesamaan_status': 'TEKS TERLALU PENDEK',
                    'total_kata': len(text.split()),
                    'kata_help_dideteksi': 0,
                    'referensi_ditemukan': 0,
                    'waktu_analisis': '0 detik',
                    'skor_keunikan': '5 dari 5',
                    'tanggal_upload': datetime.now().isoformat(),
                    'waktu_proses': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'teks_asli': text[:500] + ('...' if len(text) > 500 else ''),
                    'plagiat_detected': []
                }
            
            # Check jika ada referensi
            if self.reference_vectors is None or len(self.reference_docs) == 0:
                logger.info("ℹ️ Tidak ada referensi untuk dibandingkan")
                return {
                    'success': True,
                    'message': 'Belum ada referensi untuk dibandingkan',
                    'judul_skripsi': filename.replace('.pdf', ''),
                    'tingkat_kesamaan': 0,
                    'kesamaan_status': 'BELUM ADA REFERENSI',
                    'total_kata': len(text.split()),
                    'kata_help_dideteksi': 0,
                    'referensi_ditemukan': 0,
                    'referensi_total': 0,
                    'waktu_analisis': '0 detik',
                    'skor_keunikan': '5 dari 5',
                    'tanggal_upload': datetime.now().isoformat(),
                    'waktu_proses': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'teks_asli': text[:500] + ('...' if len(text) > 500 else ''),
                    'plagiat_detected': []
                }
            
            logger.info(f"📚 Comparing with {len(self.reference_docs)} reference documents")
            
            # Hitung similarity dengan semua referensi
            input_vector = self.vectorizer.transform([processed_text])
            similarities = cosine_similarity(input_vector, self.reference_vectors)
            
            # Analisis hasil
            max_similarity = float(np.max(similarities) * 100)
            avg_similarity = float(np.mean(similarities) * 100)
            
            logger.info(f"📊 Max similarity: {max_similarity:.2f}%, Avg: {avg_similarity:.2f}%")
            
            # Deteksi plagiat detail
            plagiarism_matches = []
            threshold = 50  # Threshold untuk deteksi plagiat
            
            colors_list = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A1', '#FFC300', '#DAF7A6']
            
            for idx, similarity in enumerate(similarities[0]):
                sim_percentage = float(similarity * 100)
                if sim_percentage > threshold:
                    color = colors_list[idx % len(colors_list)]
                    
                    # Get reference info
                    ref_filename = self.reference_metadata[idx].get('filename', f'Referensi {idx+1}')
                    ref_text = self.reference_metadata[idx].get('original_text', '')
                    
                    # Simple text matching
                    original_sentences = [s.strip() for s in ref_text.split('.') if s.strip()]
                    input_sentences = [s.strip() for s in text.split('.') if s.strip()]
                    
                    # Find matching sentences (simplified)
                    matched_sentences = []
                    for sent in input_sentences[:5]:  # Check first 5 sentences
                        if any(sent.lower() in ref_sent.lower() for ref_sent in original_sentences[:5]):
                            matched_sentences.append(sent[:100])
                    
                    plagiarism_matches.append({
                        'referensi': ref_filename,
                        'similarity': sim_percentage,
                        'color': color,
                        'matched_sentences': matched_sentences[:3],
                        'reference_excerpt': ref_text[:200] + ('...' if len(ref_text) > 200 else '')
                    })
            
            # Hitung statistik
            total_words = len(text.split())
            plagiarized_words = int(total_words * (max_similarity / 100))
            
            # Tentukan status
            if max_similarity > 50:
                status = "PLAGIAT TINGGI"
            elif max_similarity > 25:
                status = "PLAGIAT SEDANG"
            elif max_similarity > 10:
                status = "PLAGIAT RENDAH"
            else:
                status = "DOKUMEN UNIK"
            
            # Hitung skor keunikan (1-5)
            uniqueness_score = max(1, 5 - (max_similarity / 20))
            
            # Waktu analisis
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'message': 'Analisis berhasil',
                'judul_skripsi': filename.replace('.pdf', ''),
                'tingkat_kesamaan': round(max_similarity, 2),
                'rata_rata_kesamaan': round(avg_similarity, 2),
                'kesamaan_status': status,
                'total_kata': total_words,
                'kata_help_dideteksi': plagiarized_words,
                'referensi_ditemukan': len(plagiarism_matches),
                'referensi_total': len(self.reference_docs),
                'waktu_analisis': f"{analysis_time:.2f} detik",
                'skor_keunikan': f"{uniqueness_score:.1f} dari 5",
                'tanggal_upload': datetime.now().isoformat(),
                'waktu_proses': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'teks_asli': text[:1000] + ('...' if len(text) > 1000 else ''),
                'plagiat_detected': plagiarism_matches,
                'similarity_breakdown': [
                    {'referensi': self.reference_metadata[i].get('filename', f'Ref {i+1}'), 
                     'similarity': float(similarities[0][i] * 100)}
                    for i in range(len(self.reference_docs))
                ]
            }
            
            # Simpan hasil
            self.save_analysis_result(result, filename)
            
            logger.info(f"✅ Analysis completed in {analysis_time:.2f} seconds")
            logger.info(f"📈 Result: {status} ({max_similarity:.2f}% similarity)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Plagiarism check error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Gagal menganalisis plagiarisme: {str(e)}")
    
    def save_model(self):
        """Simpan model ke disk"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'reference_docs': self.reference_docs,
                'reference_vectors': self.reference_vectors,
                'reference_metadata': self.reference_metadata
            }
            
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, "plagiarism_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ Model saved to {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
    
    def save_analysis_result(self, result: Dict, filename: str):
        """Simpan hasil analisis ke file JSON"""
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            result_id = str(uuid.uuid4())[:8]
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '-', '_')).rstrip()
            result_file = os.path.join(RESULTS_DIR, f"result_{result_id}_{safe_filename}.json")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Analysis result saved: {result_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save analysis result: {e}")

# Lifespan manager untuk mengganti on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global detector
    logger.info("🚀 Starting SkripsiCheck Pro API...")
    
    # Buat direktori jika belum ada
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize detector
    detector = PlagiarismDetector()
    
    logger.info(f"✅ Model loaded: {detector.vectorizer is not None}")
    logger.info(f"✅ References count: {len(detector.reference_docs)}")
    logger.info(f"✅ Documents directory: {DOCUMENTS_DIR}")
    logger.info(f"✅ Results directory: {RESULTS_DIR}")
    
    # Cek jika ada file hasil lama, bisa dibersihkan
    try:
        results = os.listdir(RESULTS_DIR)
        if len(results) > 50:  # Jika lebih dari 50 file, hapus yang lama
            results.sort(key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)))
            for old_file in results[:-50]:
                os.remove(os.path.join(RESULTS_DIR, old_file))
            logger.info(f"🧹 Cleared old result files, kept 50 most recent")
    except Exception as e:
        logger.error(f"❌ Error cleaning old results: {e}")
    
    yield
    # Shutdown
    logger.info("🛑 Shutting down SkripsiCheck Pro API...")

# Initialize FastAPI app dengan lifespan
app = FastAPI(
    title="SkripsiCheck Pro API",
    description="API untuk deteksi plagiarisme berbasis referensi",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SkripsiCheck Pro API v2.0",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "check_plagiarism": "POST /api/plagiarism-check",
            "check_plagiarism_alt": "POST /plagiarism/check (legacy)",
            "add_reference": "POST /api/add-reference",
            "download_pdf": "GET /api/download-pdf",
            "download_report": "GET /api/download-report",
            "list_references": "GET /api/references",
            "health": "GET /api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector.vectorizer is not None,
        "references_count": len(detector.reference_docs),
        "model_type": "TF-IDF with Cosine Similarity",
        "features": {
            "max_features": 5000,
            "ngram_range": "(1, 3)",
            "stopwords": "Indonesian/English"
        }
    }

@app.post("/api/plagiarism-check")
async def check_plagiarism_endpoint(file: UploadFile = File(...)):
    """Endpoint utama untuk check plagiarisme"""
    try:
        logger.info(f"📤 Processing file: {file.filename}")
        
        # Validasi file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Hanya file PDF yang didukung")
        
        # Simpan file sementara
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        try:
            content = await file.read()
            if len(content) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="File terlalu besar. Maksimal 10MB")
            
            temp_file.write(content)
            temp_file.close()
            
            # Extract text dari PDF
            text = detector.extract_text_from_pdf(temp_file.name)
            
            if not text or len(text.strip()) < 100:
                raise HTTPException(status_code=400, detail="File PDF tidak mengandung teks yang cukup (minimal 100 karakter)")
            
            # Check plagiarism
            result = detector.check_plagiarism(text, file.filename)
            
            # Clean up
            os.unlink(temp_file.name)
            
            return result
            
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in plagiarism check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ✅ ENDPOINT LEGACY untuk compatibility dengan frontend lama
@app.post("/plagiarism/check")
async def check_plagiarism_legacy(file: UploadFile = File(...)):
    """Legacy endpoint untuk compatibility"""
    logger.info(f"🔗 Legacy endpoint called: /plagiarism/check for {file.filename}")
    return await check_plagiarism_endpoint(file)

@app.post("/api/add-reference")
async def add_reference_document(file: UploadFile = File(...)):
    """Tambah dokumen referensi"""
    try:
        logger.info(f"📥 Adding reference: {file.filename}")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Hanya file PDF yang didukung")
        
        # Simpan file
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '-', '_')).rstrip()
        saved_path = os.path.join(DOCUMENTS_DIR, f"ref_{file_hash}_{safe_filename}")
        
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File terlalu besar. Maksimal 10MB")
        
        with open(saved_path, 'wb') as f:
            f.write(content)
        
        # Metadata
        metadata = {
            'id': file_hash,
            'original_filename': file.filename,
            'file_size': len(content),
            'upload_date': datetime.now().isoformat()
        }
        
        # Tambah ke model
        success = detector.add_reference_document(saved_path, file.filename, metadata)
        
        if success:
            return {
                "success": True,
                "message": "Dokumen referensi berhasil ditambahkan",
                "reference_id": file_hash,
                "filename": file.filename,
                "references_count": len(detector.reference_docs)
            }
        else:
            raise HTTPException(status_code=500, detail="Gagal memproses dokumen referensi")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding reference: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/references")
async def list_references():
    """Dapatkan daftar referensi"""
    try:
        references = []
        for meta in detector.reference_metadata:
            references.append({
                'id': meta.get('id', 'unknown'),
                'filename': meta.get('filename', 'unknown'),
                'upload_date': meta.get('added_date', 'unknown'),
                'text_length': meta.get('processed_length', 0),
                'file_size': meta.get('file_size', 0)
            })
        
        return {
            "success": True,
            "count": len(references),
            "references": references
        }
    except Exception as e:
        logger.error(f"❌ Error listing references: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/download-pdf")
async def download_pdf_report(analysis_id: str = None, title: str = "Laporan Plagiarisme"):
    """Download hasil analisis sebagai PDF"""
    try:
        # Cari file hasil berdasarkan ID atau gunakan contoh
        result_file = None
        if analysis_id:
            for file in os.listdir(RESULTS_DIR):
                if analysis_id in file and file.endswith('.json'):
                    result_file = os.path.join(RESULTS_DIR, file)
                    break
        
        result_data = None
        if result_file and os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            logger.info(f"📄 Using saved result file: {result_file}")
        else:
            # Data contoh jika tidak ditemukan
            logger.info("📝 Using example data for PDF")
            result_data = {
                "judul_skripsi": title or "Dokumen Skripsi",
                "tingkat_kesamaan": 15.5,
                "kesamaan_status": "DOKUMEN UNIK",
                "total_kata": 2500,
                "kata_help_dideteksi": 375,
                "referensi_ditemukan": 3,
                "waktu_analisis": "45 detik",
                "skor_keunikan": "4.2 dari 5",
                "tanggal_upload": datetime.now().isoformat(),
                "waktu_proses": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "plagiat_detected": [
                    {
                        "referensi": "Jurnal Teknik Informatika 2023",
                        "similarity": 85.5,
                        "color": "#FF5733"
                    },
                    {
                        "referensi": "Skripsi Mahasiswa 2022",
                        "similarity": 65.2,
                        "color": "#33FF57"
                    }
                ]
            }
        
        # Buat PDF di memory
        buffer = io.BytesIO()
        
        # Setup dokumen PDF sederhana
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Header
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(colors.HexColor("#1976D2"))
        c.drawString(50, height - 50, "LAPORAN ANALISIS PLAGIARISME")
        
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        c.drawString(50, height - 80, f"Judul Dokumen: {result_data['judul_skripsi']}")
        c.drawString(50, height - 100, f"Tanggal Analisis: {result_data.get('waktu_proses', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        
        # Hasil Analisis
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 140, "HASIL ANALISIS")
        
        y_position = height - 170
        c.setFont("Helvetica", 12)
        
        results = [
            ("Tingkat Kesamaan", f"{result_data['tingkat_kesamaan']}%"),
            ("Status", result_data['kesamaan_status']),
            ("Total Kata Diperiksa", str(result_data['total_kata'])),
            ("Kata Terdeteksi Plagiat", str(result_data['kata_help_dideteksi'])),
            ("Referensi Ditemukan", str(result_data['referensi_ditemukan'])),
            ("Waktu Analisis", result_data['waktu_analisis']),
            ("Skor Keunikan", result_data['skor_keunikan'])
        ]
        
        for label, value in results:
            c.drawString(50, y_position, f"{label}:")
            c.drawString(250, y_position, value)
            y_position -= 25
        
        # Detail Plagiat
        if result_data.get('plagiat_detected') and len(result_data['plagiat_detected']) > 0:
            y_position -= 20
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y_position, "DETAIL TEMUAN PLAGIAT")
            y_position -= 30
            
            c.setFont("Helvetica", 12)
            for idx, item in enumerate(result_data['plagiat_detected'][:5]):  # Maks 5 item
                similarity = item.get('similarity', 0)
                status = "TINGGI" if similarity > 70 else "SEDANG" if similarity > 40 else "RENDAH"
                
                c.drawString(50, y_position, f"{idx+1}. {item.get('referensi', 'Unknown')}")
                c.drawString(350, y_position, f"{similarity:.1f}% ({status})")
                y_position -= 25
                
                if y_position < 50:  # Jika hampir habis halaman
                    c.showPage()
                    y_position = height - 50
                    c.setFont("Helvetica", 12)
        
        # Rekomendasi
        y_position -= 30
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_position, "REKOMENDASI")
        y_position -= 30
        
        c.setFont("Helvetica", 12)
        similarity = result_data['tingkat_kesamaan']
        
        if similarity > 50:
            recommendations = [
                "• Tingkat plagiat tinggi. Disarankan untuk menulis ulang bagian yang terdeteksi plagiat.",
                "• Periksa semua kutipan dan pastikan sudah dicantumkan dengan benar.",
                "• Gunakan parafrase untuk menyampaikan ide dengan kata-kata sendiri."
            ]
        elif similarity > 25:
            recommendations = [
                "• Tingkat plagiat sedang. Beberapa bagian perlu diperbaiki.",
                "• Parafrase ulang bagian yang memiliki kemiripan tinggi.",
                "• Tambahkan citation yang mungkin terlewat."
            ]
        else:
            recommendations = [
                "• Dokumen cukup unik. Pertahankan originalitas penulisan.",
                "• Teruskan penulisan dengan menjaga konsistensi gaya penulisan."
            ]
        
        for rec in recommendations:
            c.drawString(50, y_position, rec)
            y_position -= 20
        
        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.grey)
        c.drawString(50, 30, f"Dibuat oleh SkripsiCheck Pro • {datetime.now().strftime('%d %B %Y %H:%M')}")
        
        # Save PDF
        c.save()
        buffer.seek(0)
        
        # Return sebagai file download
        filename = f"laporan-plagiat-{result_data['judul_skripsi'].replace(' ', '-')}.pdf"
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gagal membuat PDF: {str(e)}")

@app.get("/api/download-report")
async def download_csv_report(analysis_id: str = None):
    """Download hasil analisis sebagai CSV"""
    try:
        # Cari file hasil
        result_file = None
        if analysis_id:
            for file in os.listdir(RESULTS_DIR):
                if analysis_id in file and file.endswith('.json'):
                    result_file = os.path.join(RESULTS_DIR, file)
                    break
        
        result_data = None
        if result_file and os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            logger.info(f"📄 Using saved result file: {result_file}")
        else:
            # Data contoh
            logger.info("📝 Using example data for CSV")
            result_data = {
                "judul_skripsi": "Dokumen Skripsi Contoh",
                "tingkat_kesamaan": 15.5,
                "kesamaan_status": "DOKUMEN UNIK",
                "total_kata": 2500,
                "kata_help_dideteksi": 375,
                "referensi_ditemukan": 3,
                "waktu_analisis": "45 detik",
                "skor_keunikan": "4.2 dari 5",
                "tanggal_upload": datetime.now().isoformat(),
                "plagiat_detected": [
                    {
                        "referensi": "Jurnal Teknik Informatika 2023",
                        "similarity": 85.5
                    }
                ]
            }
        
        # Buat DataFrame untuk CSV
        csv_data = []
        
        # Data utama
        csv_data.append(["LAPORAN ANALISIS PLAGIARISME"])
        csv_data.append([])
        csv_data.append(["PARAMETER ANALISIS", "NILAI"])
        csv_data.append(["Judul Dokumen", result_data['judul_skripsi']])
        csv_data.append(["Tanggal Analisis", result_data.get('waktu_proses', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))])
        csv_data.append(["Tingkat Kesamaan", f"{result_data['tingkat_kesamaan']}%"])
        csv_data.append(["Status", result_data['kesamaan_status']])
        csv_data.append(["Total Kata", result_data['total_kata']])
        csv_data.append(["Kata Terdeteksi Plagiat", result_data['kata_help_dideteksi']])
        csv_data.append(["Persentase Plagiat", f"{result_data['kata_help_dideteksi']/result_data['total_kata']*100:.2f}%" if result_data['total_kata'] > 0 else "0%"])
        csv_data.append(["Referensi Ditemukan", result_data['referensi_ditemukan']])
        csv_data.append(["Waktu Analisis", result_data['waktu_analisis']])
        csv_data.append(["Skor Keunikan", result_data['skor_keunikan']])
        csv_data.append([])
        
        # Detail plagiat
        if result_data.get('plagiat_detected') and len(result_data['plagiat_detected']) > 0:
            csv_data.append(["DETAIL TEMUAN PLAGIAT"])
            csv_data.append(["No", "Referensi", "Tingkat Kemiripan", "Status"])
            
            for idx, item in enumerate(result_data['plagiat_detected'], 1):
                similarity = item.get('similarity', 0)
                status = "TINGGI" if similarity > 70 else "SEDANG" if similarity > 40 else "RENDAH"
                csv_data.append([idx, item.get('referensi', 'Unknown'), f"{similarity:.1f}%", status])
            
            csv_data.append([])
        
        # Rekomendasi
        csv_data.append(["REKOMENDASI"])
        similarity = result_data['tingkat_kesamaan']
        
        if similarity > 50:
            recommendations = [
                "Tingkat plagiat tinggi. Disarankan untuk menulis ulang bagian yang terdeteksi plagiat.",
                "Periksa semua kutipan dan pastikan sudah dicantumkan dengan benar.",
                "Gunakan parafrase untuk menyampaikan ide dengan kata-kata sendiri."
            ]
        elif similarity > 25:
            recommendations = [
                "Tingkat plagiat sedang. Beberapa bagian perlu diperbaiki.",
                "Parafrase ulang bagian yang memiliki kemiripan tinggi.",
                "Tambahkan citation yang mungkin terlewat."
            ]
        else:
            recommendations = [
                "Dokumen cukup unik. Pertahankan originalitas penulisan.",
                "Teruskan penulisan dengan menjaga konsistensi gaya penulisan."
            ]
        
        for idx, rec in enumerate(recommendations, 1):
            csv_data.append([f"{idx}.", rec])
        
        csv_data.append([])
        csv_data.append(["Catatan:", "Laporan ini dibuat otomatis oleh SkripsiCheck Pro"])
        csv_data.append(["Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # Buat CSV di memory
        buffer = io.StringIO()
        
        # Convert ke DataFrame dan simpan
        df = pd.DataFrame(csv_data)
        df.to_csv(buffer, index=False, header=False)
        
        csv_content = buffer.getvalue()
        
        # Return sebagai file download
        filename = f"laporan-detil-{result_data['judul_skripsi'].replace(' ', '-')}.csv"
        
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8-sig')),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal membuat CSV: {str(e)}")

@app.delete("/api/references/{reference_id}")
async def delete_reference(reference_id: str):
    """Hapus dokumen referensi"""
    try:
        # Cari dan hapus dari metadata
        for i, meta in enumerate(detector.reference_metadata):
            if meta.get('id') == reference_id:
                # Hapus dari semua list
                detector.reference_metadata.pop(i)
                if i < len(detector.reference_docs):
                    detector.reference_docs.pop(i)
                
                # Update model
                if detector.vectorizer and len(detector.reference_docs) > 0:
                    detector.reference_vectors = detector.vectorizer.fit_transform(detector.reference_docs)
                elif len(detector.reference_docs) == 0:
                    detector.reference_vectors = None
                
                # Save model
                detector.save_model()
                
                # Hapus file fisik jika ada
                for file in os.listdir(DOCUMENTS_DIR):
                    if reference_id in file:
                        file_path = os.path.join(DOCUMENTS_DIR, file)
                        os.remove(file_path)
                        logger.info(f"🗑️ Deleted file: {file_path}")
                
                return {
                    "success": True,
                    "message": f"Referensi {reference_id} berhasil dihapus",
                    "references_count": len(detector.reference_docs)
                }
        
        raise HTTPException(status_code=404, detail="Referensi tidak ditemukan")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting reference: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============== DEMO ENDPOINTS ==============

@app.post("/api/demo/add-sample-references")
async def add_sample_references():
    """Tambahkan beberapa referensi sample untuk demo"""
    try:
        sample_texts = [
            "Machine learning adalah cabang dari kecerdasan buatan yang memungkinkan komputer belajar dari data tanpa pemrograman eksplisit.",
            "Deep learning merupakan subset dari machine learning yang menggunakan jaringan saraf tiruan dengan banyak lapisan.",
            "Natural language processing adalah bidang AI yang berfokus pada interaksi antara komputer dan bahasa manusia.",
            "Plagiarisme adalah tindakan menggunakan karya orang lain tanpa memberikan atribusi yang tepat.",
            "Deteksi plagiarisme penting dalam dunia akademik untuk memastikan orisinalitas karya ilmiah."
        ]
        
        added_count = 0
        for i, text in enumerate(sample_texts):
            # Simpan sebagai file PDF dummy
            filename = f"sample_reference_{i+1}.txt"
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Tambah sebagai referensi
            metadata = {
                'id': f"sample_{i+1}",
                'original_filename': filename,
                'file_size': len(text),
                'upload_date': datetime.now().isoformat()
            }
            
            # Simpan teks langsung (karena bukan PDF)
            detector.reference_docs.append(detector.preprocess_text(text))
            detector.reference_metadata.append({
                'filename': filename,
                'original_text': text,
                'processed_length': len(text),
                'added_date': datetime.now().isoformat(),
                **metadata
            })
            
            added_count += 1
        
        # Update model
        if detector.vectorizer and len(detector.reference_docs) > 0:
            detector.reference_vectors = detector.vectorizer.fit_transform(detector.reference_docs)
        
        detector.save_model()
        
        return {
            "success": True,
            "message": f"{added_count} referensi sample berhasil ditambahkan",
            "references_count": len(detector.reference_docs)
        }
        
    except Exception as e:
        logger.error(f"❌ Error adding sample references: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal menambahkan sample: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )