# NLP Preprocessing, Modeling, and Evaluation Comparison

## Overview

This repository contains a comprehensive Jupyter notebook for **fake news detection** using the ISOT Fake News Dataset. The project implements a complete machine learning pipeline with advanced NLP techniques, feature engineering, interpretability analysis, and comparative model evaluation.

### Key Focus Areas
- **Entity Masking**: Replacing named entities before text preprocessing
- **Feature Importance Analysis**: Extracting top 5 most important engineered features using Random Forest
- **SHAP Interpretability**: Explaining model decisions using SHapley Additive exPlanations
- **Comparative Modeling**: 27 different model configurations evaluated
- **Multiple Representations**: BoW, TF-IDF, Word2Vec, and Transformers (BERT)

---

## Project Structure

```
AI PROG NU/
├── modeling_comparison.ipynb          # Main notebook with full pipeline
├── preprocessing_pipeline.ipynb       # Data preprocessing and feature engineering
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── True.csv                           # Dataset: Real news articles
├── Fake.csv                           # Dataset: Fake news articles
├── model_performance_metrics.csv      # Comparison results of all models
│
├── data/
│   ├── train_processed.csv            # Processed training dataset
│   └── test_processed.csv             # Processed testing dataset
│
└── features/
    ├── knowledge_features_train.csv   # Engineered features for training data
    ├── knowledge_features_test.csv    # Engineered features for testing data
    ├── word_to_idx.json               # Word to index mapping
    ├── word2vec_embedding.npy         # Word2Vec embeddings
    ├── word2vec_train_padded.npy      # Padded Word2Vec embeddings (training)
    └── word2vec_test_padded.npy       # Padded Word2Vec embeddings (testing)
```

---

## Dataset Information

### Source
- **Dataset**: ISOT Fake News Dataset
- **Real News**: `True.csv`
- **Fake News**: `Fake.csv`

### Data Split
- **Training Set**: 7,500 balanced samples (3,750 Real, 3,750 Fake)
- **Testing Set**: 1,500 balanced samples (750 Real, 750 Fake)

### Dataset Features
Typical columns in the CSV files include:
- `title`: Article headline
- `text`: Article content
- `subject`: Article category/subject
- `date`: Publication date
- `label`: Real or Fake designation

---

## Notebook Structure

The main notebook (`modeling_comparison.ipynb`) is organized into the following sections:

### 1. **Data Loading and Exploration**
   - Load True.csv and Fake.csv
   - Clean and preprocess text
   - Deduplicate records
   - Verify balanced splits

### 2. **Dataset Sampling and Split**
   - Create balanced training and testing sets
   - Explore sentence length distributions
   - Generate visualizations

### 3. **Entity Masking**
   - Replace named entities with tags (e.g., `[PERSON]`, `[ORG]`, `[GPE]`)
   - Compare masked vs. unmasked text
   - Analyze entity distributions

### 4. **Feature Engineering**
   - Extract linguistic features (word count, sentence count, etc.)
   - Sentiment analysis
   - POS (Part-of-Speech) tagging counts
   - NER (Named Entity Recognition) counts
   - Additional domain-specific features

### 5. **Feature Importance Analysis**
   - Train Random Forest on engineered features
   - Extract and visualize top 5 most important features
   - Analyze feature contributions to classification

### 6. **Text Representations**
   - **Bag of Words (BoW)**: Simple word frequency vectors
   - **TF-IDF**: Term Frequency-Inverse Document Frequency
   - **Word2Vec**: Dense word embeddings
   - **BERT**: Transformer-based contextual embeddings

### 7. **SHAP Interpretability**
   - Calculate SHAP values for model explainability
   - Visualize feature contributions
   - Generate force plots and dependence plots

### 8. **Model Training**
   - Rule-based classifiers
   - Logistic Regression
   - Linear SVM
   - Naïve Bayes
   - Random Forest
   - Deep Learning (Neural Networks)
   - Transformers (BERT)

### 9. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - ROC-AUC curves
   - Comparative analysis across all 27 configurations

### 10. **Visualizations**
   - Consolidated score comparison table
   - 7 separate score bar plots with embedded text values
   - 4 separate confusion matrix blocks
   - Feature importance charts
   - SHAP summary plots

---

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~2GB for models and embeddings

### Environment Setup
Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv1

# Activate virtual environment
# On Windows:
.\.venv1\Scripts\Activate.ps1

# On macOS/Linux:
source .venv1/bin/activate
```

---

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

### 3. Download Spacy Model
```bash
python -m spacy download en_core_web_sm
```

---

## Running the Notebook

### Option 1: Jupyter Lab/Notebook
```bash
# Launch Jupyter
jupyter notebook modeling_comparison.ipynb

# Or use Jupyter Lab
jupyter lab modeling_comparison.ipynb
```

### Option 2: VS Code
1. Open VS Code
2. Install the "Jupyter" extension
3. Open the notebook file
4. Click "Run All" or run cells individually

### Running Specific Sections
- Execute cells sequentially from top to bottom
- Each section builds on the previous one
- Ensure data files (True.csv, Fake.csv) are in the working directory

---

## Key Features

### 1. **Entity Masking**
Protects privacy and tests model robustness by replacing named entities:
```python
John Smith → [PERSON]
Google → [ORG]
New York → [GPE]
```

### 2. **Comparative Analysis**
27 model configurations tested:
- **3 Base Models**: Rule-based, Traditional ML, Deep Learning, Transformers
- **4 Text Representations**: BoW, TF-IDF, Word2Vec, BERT
- **2 Feature Sets**: With and without top 5 engineered features

### 3. **Interpretability**
- Feature importance rankings
- SHAP value explanations
- Model decision visualization

### 4. **Comprehensive Evaluation**
- Multiple evaluation metrics
- Confusion matrices for each model
- Side-by-side performance comparison

---

## Output Files

The notebook generates the following output files:

| File Name | Description |
|-----------|-------------|
| `data/train_processed.csv` | Cleaned and processed training data |
| `data/test_processed.csv` | Cleaned and processed testing data |
| `features/knowledge_features_train.csv` | Engineered features for training set |
| `features/knowledge_features_test.csv` | Engineered features for testing set |
| `features/word_to_idx.json` | Word to index vocabulary mapping |
| `features/word2vec_embedding.npy` | Word2Vec embeddings array |
| `features/word2vec_train_padded.npy` | Padded embeddings for training |
| `features/word2vec_test_padded.npy` | Padded embeddings for testing |
| `model_performance_metrics.csv` | Final comparison of all 27 models |

---

## Dependencies

### Core Libraries
- **pandas** (≥2.0.0): Data manipulation
- **numpy** (≥1.20.0): Numerical computing
- **scikit-learn** (≥1.2.0): Machine learning algorithms
- **nltk** (≥3.8.0): Natural Language Toolkit
- **spacy** (≥3.5.0): Advanced NLP

### Deep Learning & Embeddings
- **torch** (≥2.0.0): PyTorch framework
- **transformers** (≥4.28.0): BERT and other transformers
- **beautifulsoup4** (≥4.11.0): HTML parsing

### Additional Tools
- **networkx** (≥3.0): Graph analysis
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization
- **shap**: Model interpretability

---

## Notes

### Word2Vec Compilation
- **Windows**: Requires MSVC compiler (Visual C++ Build Tools)
- If compilation fails, the notebook includes a `MockWord2Vec` fallback
- To avoid issues, use pre-built wheels or the fallback

### Memory Considerations
- Large datasets may require significant RAM
- BERT embeddings are memory-intensive
- Consider reducing dataset size for limited memory systems

### Reproducibility
- Fixed random seed (`RANDOM_STATE = 42`)
- Results should be reproducible across runs
- Model training may show slight variations due to randomness in neural networks

---

## Performance Metrics

The notebook evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Confusion Matrix**: Breakdown of true/false positives and negatives

---

## Example Usage

```python
# Load processed data
train_data = pd.read_csv('data/train_processed.csv')
test_data = pd.read_csv('data/test_processed.csv')

# Load engineered features
train_features = pd.read_csv('features/knowledge_features_train.csv')
test_features = pd.read_csv('features/knowledge_features_test.csv')

# Load embeddings
embeddings = np.load('features/word2vec_embedding.npy')
train_vectors = np.load('features/word2vec_train_padded.npy')
test_vectors = np.load('features/word2vec_test_padded.npy')

# Load results
results = pd.read_csv('model_performance_metrics.csv')
print(results)
```

---

## Troubleshooting

### Issue: Module Not Found
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: CUDA/GPU Not Available
**Solution**: CPU mode works fine; PyTorch will automatically use CPU
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Word2Vec Compilation Error
**Solution**: Install Visual C++ Build Tools or use the fallback
- Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Or re-run: The notebook will auto-switch to MockWord2Vec

### Issue: Memory Error
**Solution**: Reduce dataset size or run on a machine with more RAM
```python
# Modify sampling parameters in the notebook
TRAIN_SIZE = 5000  # Instead of 7500
TEST_SIZE = 1000   # Instead of 1500
```

---

## Future Enhancements

- [ ] Add ensemble methods combining multiple models
- [ ] Implement cross-validation for robust evaluation
- [ ] Add real-time inference functionality
- [ ] Develop API endpoint for model serving
- [ ] Extend to multilingual fake news detection
- [ ] Add temporal analysis of news articles

---

## License

This project is provided as-is for academic and research purposes.

---

## Contact & Support

For issues, questions, or suggestions, please refer to the notebook documentation and comments within the code.

---

## Citation

If you use this project in your research, please cite:

```
NLP Preprocessing, Modeling, and Evaluation Comparison
Fake News Detection using ISOT Dataset
Academic Framework Implementation
```

---

**Last Updated**: 2026-08-26  
**Status**: Active  
**Version**: 1.0
