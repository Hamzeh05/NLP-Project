# Arabic Text Summarization — NLP Project

An abstractive Arabic text summarization system built using a Seq2Seq architecture with Bahdanau Attention, implemented in PyTorch, complete with a locally deployable Streamlit web interface.

---

# Project Structure

```text
NLP-Project/
│
├── Phase 1 (Data loading & Data Preprocessing).ipynb
├── Phase 2 (Model Design, Training & Evaluation).ipynb
├── app.py                      # Streamlit Web UI for model inference
├── checkpoint.pt               # Trained PyTorch model weights
├── tokenizer.pkl               # Saved tokenizer vocabulary
├── NLP_Project_Report.pdf      # Detailed project documentation and findings
├── requirements.txt
└── README.md
```

---

# Phase 1 — Data Loading & Preprocessing

## Data Aggregation & Filtering

We merged four distinct Arabic datasets into a single master dataframe. Duplicate entries were removed and null rows were dropped to ensure a clean and high-quality dataset.

| Dataset | Source | Rows |
|---|---|---|
| SumArabic | Common Crawl (downloaded via scripts) | ~2,600 |
| AraSum | UFAL DSG | ~49,000 |
| Egyptian | HuggingFace | ~3,689 |
| Kaggle | Kaggle | ~8,055 |

---

## Exploratory Data Analysis (EDA)

EDA was used to guide all preprocessing decisions.

### Performed Analysis

- Distribution analysis for document and summary lengths
- Anomaly detection for invalid summary/document ratios
- Threshold selection for sequence truncation and padding

### Final Sequence Lengths

```python
MAX_ENC_LEN = 400
MAX_DEC_LEN = 50
```

Documents are padded/truncated to 400 tokens, while summaries are padded/truncated to 50 tokens.

---

## Text Cleaning & Normalization

The `clean_arabic()` function standardizes Arabic text through multiple preprocessing stages:

### Cleaning Operations

- Remove URLs
- Remove HTML tags and noise
- Remove English characters
- Remove punctuation
- Remove Arabic diacritics (Tashkeel)

### Character Normalization

```text
إ أ آ ا  →  ا
ى        →  ي
ة        →  ه
```

---

## Tokenization & Padding

- A shared tokenizer is trained on both documents and summaries
- Vocabulary size is limited to:

```python
VOCAB_SIZE = 40000
```

- Sequences are converted into integer token IDs
- Post-padding is applied for fixed-length tensors
- Target summaries include:
  - `sostok`
  - `eostok`

---

# Phase 2 — Model Design, Training & Evaluation

## Model Architecture

The system uses a Seq2Seq architecture with Bahdanau Attention implemented entirely in PyTorch.

### Core Dimensions

```python
LATENT_DIM = 256
EMBEDDING_DIM = 300
VOCAB_SIZE = 40000
```

---

## 1. Shared Embedding Layer

A single embedding matrix is shared between the encoder and decoder.

### Embedding Shape

```python
40000 × 300
```

This ensures that identical Arabic words share the same semantic representation on both sides of the Seq2Seq pipeline.

---

## 2. Encoder — Bidirectional LSTM

### Encoder Configuration

- 2-layer Bidirectional LSTM
- Input size: `300`
- Hidden size: `256` per direction
- Dropout: `0.3`

### Output Representation

Forward and backward hidden states are concatenated:

```python
256 + 256 = 512
```

This produces a contextual encoder representation of size `512`.

---

## 3. Bahdanau Attention Mechanism

At each decoding timestep, Bahdanau Attention computes alignment scores between:

- Current decoder hidden state
- All encoder outputs

### Attention Components

```python
W1
W2
V
```

The resulting weighted context vector allows the decoder to focus dynamically on the most relevant parts of the input document.

Context vector size:

```python
512
```

---

## 4. Decoder — Unidirectional LSTM

### Decoder Input

The decoder input consists of:

```python
300  → Embedded previous token
512  → Attention context vector
```

Combined input size:

```python
812
```

### Decoder Configuration

- Unidirectional LSTM
- Hidden size: `512`

### Output Classifier

The decoder output is passed through two fully connected layers:

```python
1024 → 512 → 40000
```

with:

- ReLU activation
- Dropout (`0.3`)

to predict the next token in the vocabulary.

---

# Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 3×10⁻⁴ |
| Loss Function | CrossEntropyLoss (label smoothing = 0.1) |
| Gradient Clipping | 5.0 |
| Batch Size | 64 |
| Scheduler | OneCycleLR |
| Early Stopping | Patience = 5 |
| Mixed Precision | AMP (float16) |
| Teacher Forcing | Enabled |

---

# Evaluation Results

| Metric | Score |
|---|---|
| ROUGE-1 | Insert S |
| ROUGE-2 | Insert Score |
| ROUGE-L | Insert Score |
| BERTScore Precision | 0.3911 |
| BERTScore Recall | 0.2959 |
| BERTScore F1 | 0.3345 |

---

# Interactive Web Interface (Streamlit)

The repository includes a lightweight Streamlit interface (`app.py`) for real-time Arabic summarization.

The interface loads:

- `checkpoint.pt`
- `tokenizer.pkl`

and performs live abstractive summarization directly in the browser.

---

## Streamlit Features

- Real-time Arabic text cleaning
- Greedy decoding inference
- Dynamic minimum-length penalty
- Prevention of premature `<eostok>` generation
- Fully local deployment

---

# Setup & Usage

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run the Web Interface

Ensure the following files exist in the same directory:

```text
checkpoint.pt
tokenizer.pkl
app.py
```

Then launch Streamlit:

```bash
streamlit run app.py
```

The application will automatically open in your default browser.

---

## 3. Retraining on Google Colab

To retrain the model from scratch:

1. Open the notebooks in Google Colab
2. Enable GPU runtime:

```text
Runtime → Change runtime type → T4 GPU
```

3. Upload the datasets to `/content/`
4. Run:
   - Phase 1 notebook first
   - Phase 2 notebook second

---

# Datasets

| Dataset | Download |
|---|---|
| SumArabic | Run `downloader.py` from the official zip |
| AraSum | https://github.com/UFAL-DSG/sumarabic |
| Egyptian | https://huggingface.co/datasets/Omar-youssef/Egyptian-text-summarization |
| Kaggle | https://www.kaggle.com/datasets/haithemhermessi/arabic-news-summarization |

---

## Required Dataset Files

```text
sumarabic-1.0-train.jsonl
sumarabic-1.0-test.jsonl
sumarabic-1.0-valid.jsonl
AraSum.txt
train-00000-of-00001.parquet
summarizdataset.csv
```

---

# Requirements

See `requirements.txt` for the complete dependency list.

## Main Libraries

| Library | Purpose |
|---|---|
| torch | Model training and inference |
| streamlit | Web interface |
| transformers | BERTScore evaluation |
| bert-score | Semantic similarity evaluation |
| rouge-score | ROUGE evaluation |
| tensorflow | Tokenization and preprocessing |

---

# Notes

- TensorFlow is used only for:
  - `Tokenizer`
  - `pad_sequences`

- All model training and inference are implemented purely in PyTorch.

- Training on a Google Colab T4 GPU typically takes:

```text
30–60 minutes
```

depending on dataset size.

- Model checkpoints are automatically saved whenever validation loss improves.

---
