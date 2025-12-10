
# PLAGIARISM DETECTION MODELS

## Model Details
- Created: 2025-08-20 16:36:53.071852
- Number of documents: 26
- TF-IDF Dimensions: (26, 7047)
- Vocabulary Size: 7047

## Files:
1. tfidf_vectorizer.pkl - TF-IDF vectorizer model
2. preprocessing_artifacts.joblib - Preprocessing tools
3. document_database.csv - Document metadata
4. tfidf_matrix_sparse.npz - TF-IDF matrix (sparse format)
5. similarity_matrix.npy - Cosine similarity matrix
6. similarity_dataframe.csv - Similarity results with labels
7. vocabulary.npy - Feature names
8. predict_function.pkl - Prediction function

## Usage:
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tv = pickle.load(f)

# Load predict function
with open('predict_function.pkl', 'rb') as f:
    predict_similarity = pickle.load(f)

# Predict similarity
results = predict_similarity("teks dokumen baru", threshold=0.3)
