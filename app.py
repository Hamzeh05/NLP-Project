# cd "C:\Users\hamze\OneDrive\Desktop\NLP Project"
#python -m streamlit run app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import pickle
import re
import numpy as np
from collections import Counter

MAX_ENC_LEN = 400
MAX_DEC_LEN = 50
VOCAB_SIZE = 40000
LATENT_DIM = 256
EMBEDDING_DIM = 300

device = torch.device("cpu")

TOKENIZER_PATH = r"c:\Users\hamze\OneDrive\Desktop\NLP Project\tokenizer.pkl"
CHECKPOINT_PATH = r"c:\Users\hamze\OneDrive\Desktop\NLP Project\checkpoint.pt"


class SimpleTokenizer:
    def __init__(self, num_words=40000, oov_token="<UNK>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()

    def texts_to_sequences(self, texts):
        sequences = []

        for text in texts:
            seq = [self.word_index.get(word, 1) for word in str(text).split()]
            sequences.append(seq)

        return sequences


def clean_arabic(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|[a-zA-Z]", "", text)
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)

    text = re.sub(r"([^\w\s\ا-ي0-9\.،؟])", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def pad_sequences(sequences, maxlen=None, padding="post", value=0):
    num_samples = len(sequences)

    if maxlen is None:
        maxlen = max(len(s) for s in sequences) if sequences else 0

    padded = np.full((num_samples, maxlen), value, dtype=np.int64)

    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue

        trunc = seq[:maxlen] if len(seq) > maxlen else seq

        if padding == "post":
            padded[i, :len(trunc)] = trunc
        else:
            padded[i, -len(trunc):] = trunc

    return padded


shared_emb = nn.Embedding(
    VOCAB_SIZE,
    EMBEDDING_DIM,
    padding_idx=0,
)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = shared_emb

        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            LATENT_DIM,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3,
        )

    def _lstm_forward(self, emb):
        return self.lstm(emb)

    def forward(self, x):
        emb = self.embedding(x)

        outputs, (h, c) = torch.utils.checkpoint.checkpoint(
            self._lstm_forward,
            emb,
            use_reentrant=False,
        )

        h = torch.cat([h[-2], h[-1]], dim=1)
        c = torch.cat([c[-2], c[-1]], dim=1)

        return outputs, h, c


class BahdanauAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.W1 = nn.Linear(LATENT_DIM * 2, LATENT_DIM * 2)
        self.W2 = nn.Linear(LATENT_DIM * 2, LATENT_DIM * 2)
        self.V = nn.Linear(LATENT_DIM * 2, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        decoder_hidden = decoder_hidden.unsqueeze(1)

        score = self.V(
            torch.tanh(
                self.W1(encoder_outputs) +
                self.W2(decoder_hidden)
            )
        )

        if mask is not None:
            score = score.masked_fill(
                mask.unsqueeze(-1) == 0,
                -1e9,
            )

        attention_weights = torch.softmax(score, dim=1)

        context = (attention_weights * encoder_outputs).sum(dim=1)

        return context


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = shared_emb

        self.attention = BahdanauAttention()

        self.lstm = nn.LSTM(
            EMBEDDING_DIM + LATENT_DIM * 2,
            LATENT_DIM * 2,
            batch_first=True,
            dropout=0.3,
        )

        self.fc1 = nn.Linear(LATENT_DIM * 4, LATENT_DIM * 2)
        self.fc2 = nn.Linear(LATENT_DIM * 2, VOCAB_SIZE)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    # IMPORTANT: mask=None is REQUIRED
    def forward(
        self,
        x,
        h,
        c,
        encoder_outputs,
        mask=None,
    ):

        emb = self.embedding(x)

        context = self.attention(
            h.squeeze(0),
            encoder_outputs,
            mask,
        ).unsqueeze(1)

        lstm_input = torch.cat(
            [emb, context],
            dim=2,
        )

        out, (h, c) = self.lstm(
            lstm_input,
            (h, c),
        )

        fc_input = torch.cat(
            [out.squeeze(1), context.squeeze(1)],
            dim=1,
        )

        pred = self.fc2(
            self.dropout(
                self.relu(
                    self.fc1(fc_input)
                )
            )
        )

        return pred, h, c

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, trg):
        raise NotImplementedError(
            "Training forward pass is not used in this app."
        )


@st.cache_resource
def load_model_and_tokenizer():

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    model = Seq2Seq()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, tokenizer


def get_special_token_ids(tokenizer):

    word_index = tokenizer.word_index

    sos_idx = word_index.get("sostok")
    eos_idx = word_index.get("eostok")
    unk_idx = word_index.get("<UNK>", 1)

    if sos_idx is None:
        raise ValueError(
            "Tokenizer is missing the start token: sostok"
        )

    if eos_idx is None:
        raise ValueError(
            "Tokenizer is missing the end token: eostok"
        )

    return sos_idx, eos_idx, unk_idx


def make_extractive_summary(text, max_words=38):
    cleaned_text = clean_arabic(text)
    sentences = [
        part.strip()
        for part in re.split(r"[.!؟،\n]", cleaned_text)
        if len(part.strip().split()) >= 5
    ]

    if sentences:
        summary = sentences[0]
    else:
        summary = cleaned_text

    return " ".join(summary.split()[:max_words]).strip()


def clean_generated_summary(text):
    text = clean_arabic(text)
    words = text.split()
    cleaned = []

    for word in words:
        if len(cleaned) >= 2 and cleaned[-1] == word and cleaned[-2] == word:
            continue
        cleaned.append(word)

    return " ".join(cleaned).strip()


def source_overlap_ratio(summary, source_text):
    stopwords = {
        "في", "من", "علي", "على", "الي", "إلى", "الى", "ان", "أن", "إن",
        "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "كان", "كانت", "كما",
        "بعد", "قبل", "التي", "الذي", "الذين", "انه", "أنها", "او", "أو",
        "و", "ثم", "قد", "تم", "لا", "ما", "لم", "لن", "هو", "هي",
        "كل", "بين", "خلال", "اليوم", "امس", "هناك", "حيث",
    }

    summary_words = [
        word for word in summary.split()
        if len(word) > 2 and word not in stopwords
    ]

    if not summary_words:
        return 0.0

    source_words = {
        word for word in clean_arabic(source_text).split()
        if len(word) > 2 and word not in stopwords
    }

    overlap = sum(1 for word in summary_words if word in source_words)

    return overlap / len(summary_words)


def has_repeat_ngram(words, n=3):
    if len(words) < n * 2:
        return False

    ngrams = [
        tuple(words[i:i + n])
        for i in range(len(words) - n + 1)
    ]

    return len(ngrams) != len(set(ngrams))


def decode_with_beam(
    padded_seq,
    source_text,
    model,
    tokenizer,
    min_length=8,
    beam_width=6,
    top_k=18,
    length_penalty=0.9,
    debug_lines=None,
):
    reverse_vocab = tokenizer.index_word
    sos_idx, eos_idx, unk_idx = get_special_token_ids(tokenizer)
    blocked_ids = {0, unk_idx, sos_idx}

    with torch.no_grad():
        src = torch.tensor(padded_seq, dtype=torch.long).to(device)
        src_mask = (src != 0).to(device)
        enc_out, h, c = model.encoder(src)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)

        beams = [(0.0, [sos_idx], h, c)]
        completed = []

        for step in range(MAX_DEC_LEN - 1):
            candidates = []

            for score, token_ids, bh, bc in beams:
                inp = torch.tensor([[token_ids[-1]]], dtype=torch.long).to(device)

                pred, nh, nc = model.decoder(
                    inp,
                    bh,
                    bc,
                    enc_out,
                    src_mask,
                )

                for bad_idx in blocked_ids:
                    if 0 <= bad_idx < pred.shape[1]:
                        pred[0, bad_idx] = -float("inf")

                if step < min_length and 0 <= eos_idx < pred.shape[1]:
                    pred[0, eos_idx] = -float("inf")

                log_probs = F.log_softmax(pred, dim=1)
                top_probs, top_idx = log_probs.topk(top_k)

                for k in range(top_k):
                    next_id = top_idx[0, k].item()
                    next_score = score + top_probs[0, k].item()
                    next_ids = token_ids + [next_id]

                    decoded_words = [
                        reverse_vocab.get(idx, "")
                        for idx in next_ids
                        if idx not in {sos_idx, eos_idx}
                    ]

                    decoded_words = [
                        word for word in decoded_words
                        if word not in {"", "<UNK>", "sostok", "eostok"}
                    ]

                    if has_repeat_ngram(decoded_words, n=3):
                        continue

                    if len(decoded_words) != len(set(decoded_words)):
                        next_score -= 0.35

                    candidates.append((next_score, next_ids, nh, nc))

            if not candidates:
                break

            candidates.sort(
                key=lambda item: item[0] / (max(len(item[1]) - 1, 1) ** length_penalty),
                reverse=True,
            )

            beams = []

            for candidate in candidates[:beam_width]:
                if candidate[1][-1] == eos_idx:
                    completed.append(candidate)
                else:
                    beams.append(candidate)

            if debug_lines is not None and candidates:
                best_token = reverse_vocab.get(candidates[0][1][-1], "")
                debug_lines.append(f"Step {step + 1}: best token={repr(best_token)}")

            if not beams:
                break

        final_candidates = completed if completed else beams

        if not final_candidates:
            return ""

        best = max(
            final_candidates,
            key=lambda item: item[0] / (max(len(item[1]) - 1, 1) ** length_penalty),
        )

        words = []

        for token_id in best[1][1:]:
            if token_id == eos_idx:
                break

            word = reverse_vocab.get(token_id, "")

            if word not in {"", "<UNK>", "sostok", "eostok"}:
                words.append(word)

    return clean_generated_summary(" ".join(words))


def generate_summary(
    text,
    model,
    tokenizer,
    min_length=8,
    decoding_mode="Auto",
    debug=False,
):
    cleaned_text = clean_arabic(text)

    if not cleaned_text:
        return "", ["Input became empty after cleaning."]

    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_ENC_LEN, padding="post")
    _, _, unk_idx = get_special_token_ids(tokenizer)

    total_words = len(seq[0])
    unknown_words = seq[0].count(unk_idx)
    unknown_ratio = unknown_words / max(total_words, 1)

    debug_lines = [
        f"Cleaned text: {cleaned_text}",
        f"Total words: {total_words}",
        f"Unknown words: {unknown_words}",
        f"Unknown ratio: {unknown_ratio:.2%}",
        f"First 50 input ids: {seq[0][:50]}",
        f"Decoding mode: {decoding_mode}",
    ]

    neural_summary = decode_with_beam(
        padded_seq,
        cleaned_text,
        model,
        tokenizer,
        min_length=min_length,
        debug_lines=debug_lines if debug else None,
    )

    extractive_summary = make_extractive_summary(cleaned_text)
    overlap = source_overlap_ratio(neural_summary, cleaned_text)
    neural_word_count = len(neural_summary.split())

    use_extractive = (
        decoding_mode == "Reliable extractive"
        or neural_word_count < 5
        or overlap < 0.40
        or unknown_ratio > 0.35
    )

    if decoding_mode == "Neural only":
        use_extractive = False

    if use_extractive:
        final_summary = extractive_summary
        debug_lines.append("Output source: extractive fallback")
    else:
        final_summary = neural_summary
        debug_lines.append("Output source: neural beam decoder")

    debug_lines.extend([
        f"Neural summary: {neural_summary}",
        f"Extractive fallback: {extractive_summary}",
        f"Neural/source overlap: {overlap:.2%}",
    ])

    return final_summary, debug_lines if debug else []


def main():

    st.set_page_config(
        page_title="Arabic Text Summarizer",
        page_icon="📝",
    )

    st.title("📝 Arabic NLP Text Summarizer")

    st.write(
        "Enter an Arabic document below. Auto mode uses the trained model when "
        "its output looks coherent, and falls back to a source-based summary "
        "when the model generates unrelated text."
    )

    try:
        model, tokenizer = load_model_and_tokenizer()

    except Exception as e:

        st.error(
            "⚠️ Could not load the model or tokenizer."
        )

        st.exception(e)
        st.stop()

    reference_input = st.text_area(
        "Enter your Arabic text:",
        height=200,
        placeholder="أدخل النص هنا...",
    )

    min_length = st.slider(
        "Minimum summary length",
        min_value=1,
        max_value=20,
        value=8,
    )

    decoding_mode = st.radio(
        "Generation mode",
        options=[
            "Reliable extractive",
            "Auto",
            "Neural only",
        ],
        horizontal=True,
        index=0,
        help=(
            "Reliable extractive is safest for demos. Auto uses the neural "
            "model only when its output stays grounded in the source. Neural "
            "only shows the raw model behavior."
        ),
    )

    debug = st.checkbox(
        "Show debug information"
    )

    if st.button(
        "Generate Summary",
        type="primary",
    ):

        if not reference_input.strip():
            st.warning(
                "Please enter a text document to summarize."
            )
            return

        with st.spinner("Generating summary..."):

            try:

                generated_output, debug_lines = generate_summary(
                    reference_input,
                    model,
                    tokenizer,
                    min_length=min_length,
                    decoding_mode=decoding_mode,
                    debug=debug,
                )

                st.success("Generation Complete!")

                st.subheader("Generated Summary:")

                if generated_output:
                    st.info(generated_output)

                    if decoding_mode == "Auto":
                        st.caption(
                            "Auto mode may use an extractive fallback when the "
                            "trained Seq2Seq model produces an empty or unrelated summary."
                        )

                else:
                    st.warning(
                        "The model generated an empty summary."
                    )

                if debug:

                    st.subheader("Debug Information")

                    st.code(
                        "\n".join(debug_lines),
                        language="text",
                    )

                    if debug_lines:
                        st.caption(
                            "If the unknown ratio is high, "
                            "the tokenizer is not recognizing "
                            "your input words."
                        )

            except Exception as e:

                st.error(
                    "An error happened during generation."
                )

                st.exception(e)


if __name__ == "__main__":
    main()
