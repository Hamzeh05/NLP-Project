# Arabic Text Summarization — NLP Project

An abstractive Arabic text summarization system built using a Seq2Seq architecture with Bahdanau attention, implemented in PyTorch.

---

## Project Structure

NLP-Project/
│
├── Phase 1 (Data loading & Data Preprocessing).ipynb
├── Phase 2 (Model Design, Training & Evaluation).ipynb
├── requirements.txt
└── README.md

---

## Phase 1 — Data Loading & Preprocessing

### Data Aggregation & Filtering
We merged four distinct Arabic datasets into a single master dataframe. Duplicate entries were dropped and null rows were removed to ensure a high-quality baseline.

| Dataset | Source | Rows |
|---|---|---|
| SumArabic | Common Crawl (downloaded via scripts) | ~2,600 |
| AraSum | UFAL DSG | ~49,000 |
| Egyptian | HuggingFace | ~3,689 |
| Kaggle | Kaggle | ~8,055 |

### Exploratory Data Analysis (EDA)
EDA drove all preprocessing decisions:
- **Distribution Analysis** — plotted word counts for documents and summaries
- **Anomaly Detection** — identified and removed summaries longer than their source documents (ratio > 1.0)
- **Threshold Setting** — 75% of documents are under 425 words → set `MAX_ENC_LEN = 400`; summaries are consistently 30–36 words → set `MAX_DEC_LEN = 50`

### Text Cleaning & Normalization
The `clean_arabic` function standardizes raw Arabic text:
- Removes URLs, HTML noise, and English characters
- Strips diacritics (Tashkeel)
- Normalizes character variants (e.g., إأآا → ا, ى → ي, ة → ه)

### Tokenization & Padding
- A single shared tokenizer is fitted on training documents and summaries (`VOCAB_SIZE = 60,000`)
- Sequences are converted to integer IDs and post-padded to uniform lengths
- Encoder inputs padded to shape `(N, 400)`, decoder inputs to `(N, 50)`
- Start token `sostok` and end token `eostok` are added to all summaries

---

## Phase 2 — Model Design, Training & Evaluation

### Architecture
A Seq2Seq model with three core components:

**Encoder — 2-layer Bidirectional LSTM**
Reads the input document in both directions. Forward and backward hidden states are concatenated to form a unified context representation of shape `(batch, 1024)`.

**Bahdanau Attention**
At each decoding step, computes a weighted sum over all encoder outputs — allowing the decoder to focus on the most relevant parts of the source document rather than relying solely on the final encoder state.

**Decoder — Unidirectional LSTM**
Generates the summary one token at a time. Receives the previous token, current hidden state, and attention context vector as input. Hidden dimension is `1024` (LATENT_DIM × 2) to match encoder states.

**Shared Embedding**
A single embedding table is shared across encoder and decoder. Since both source and target are Arabic, this ensures identical words have identical representations on both sides, improving attention alignment.

### Training
| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 3×10⁻⁴ |
| Loss Function | CrossEntropyLoss (label smoothing 0.1) |
| Gradient Clipping | 5.0 |
| Batch Size | 64 |
| Scheduler | OneCycleLR |
| Early Stopping | Patience = 5 |
| Mixed Precision | AMP (float16) |
| Teacher Forcing | Yes |

### Evaluation Results
| Metric | Score |
|---|---|
| ROUGE-1 | — |
| ROUGE-2 | — |
| ROUGE-L | — |
| BERTScore Precision | — |
| BERTScore Recall | — |
| BERTScore F1 | — |



---

## Setup & Usage

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Download datasets
See the dataset section below for download links. Upload all files to `/content/` in Google Colab.

### 3 — Run on Google Colab
Open the notebooks in Google Colab with **Runtime → Change runtime type → T4 GPU**.

Run Phase 1 first, then Phase 2.

---

## Datasets

| Dataset | Download |
|---|---|
| SumArabic | Run `downloader.py` (included in SumArabic zip) |
| AraSum | [UFAL DSG GitHub](https://github.com/UFAL-DSG/sumarabic) |
| Egyptian | [HuggingFace](https://huggingface.co/datasets/Omar-youssef/Egyptian-text-summarization) |
| Kaggle | [Kaggle](https://www.kaggle.com/datasets/haithemhermessi/arabic-news-summarization) |

Required files to upload to Colab:
- `sumarabic-1.0-train.jsonl`
- `sumarabic-1.0-test.jsonl`
- `sumarabic-1.0-valid.jsonl`
- `AraSum.txt`
- `train-00000-of-00001.parquet`
- `summarizdataset.csv`

---

## Requirements
See `requirements.txt` for full dependency list.

Key dependencies:
- `torch` — model training
- `transformers` — BERTScore evaluation
- `rouge-score` — ROUGE evaluation
- `bert-score` — semantic evaluation
- `tensorflow` — tokenizer (preprocessing only)

---

## Notes
- TensorFlow is used only for the `Tokenizer` and `pad_sequences` in preprocessing. All model training runs on PyTorch with full GPU support.
- Training on a Google Colab T4 GPU takes approximately 30–60 minutes depending on dataset size.
- Checkpoints are saved automatically to `checkpoint.pt` whenever validation loss improves.
