# NLP-Project
Phase 1: Data Pre_Processing Description:
1. Data Aggregation & Filtering
We successfully merged four distinct Arabic datasets (SumArabic, AraSum, Egyptian, and Kaggle) into a single master dataframe containing over 61,000 unique records. We dropped duplicate entries and removed any rows containing null values to ensure a high-quality baseline.

2. The Role of Exploratory Data Analysis (EDA)
We utilized EDA as the primary driver for our preprocessing parameters, specifically focusing on sequence lengths:

Distribution Analysis: We calculated and plotted the word counts for all documents and summaries to understand the shape of the data.

Anomaly Detection: EDA revealed logical outliers in the raw data, such as summaries that were longer than their source documents. We filtered these anomalies out of the dataset.

Setting the Threshold: By analyzing the length distributions, we identified the optimal cut-off point that captures the majority of the text without wasting memory. Based on these EDA insights, we set a hard maximum sequence length of 400 tokens for the documents.

3. Text Cleaning & Normalization
To optimize the vocabulary, we standardized the raw Arabic text. This involved stripping out irrelevant web noise (HTML tags, English characters), removing diacritics (Tashkeel), and normalizing characters (e.g., standardizing various forms of Alef to a bare Alef). This step significantly reduced the vocabulary size and prevents the model from treating identical words as distinct tokens.

4.Tokenization and Matrix Construction To translate the text into a machine-readable format, we applied tokenization and padding across our dataset splits:

Vocabulary Mapping: We built a vocabulary dictionary, mapping every unique Arabic word in the dataset to a specific integer ID.

Sequencing: We converted each document and summary into a sequence of these integer IDs across the training, validation, and testing sets.

Padding to Shape (N, 400) and (N, 50): Implementing the thresholds discovered during EDA, we standardized our matrices using post-padding. The encoder inputs (documents) were padded with zeros to exactly 400 positions, while the decoder inputs (summaries) were padded to 50 positions. This was applied systematically across all three data splits, resulting in final, uniform input matrices fully prepared for the Seq2Seq architecture.




**NLP Project
Phase 2: Model Design, Training & Evaluation**

**Model Design & Architecture**
We implemented a Seq2Seq architecture built entirely in PyTorch to leverage native Windows GPU acceleration, as TensorFlow 2.11+ does not support GPU on native Windows. The architecture consists of three core components working in sequence.

The Encoder is a Bidirectional LSTM that reads the full input document in both directions simultaneously. By processing the sequence forward and backward, it captures contextual dependencies from both sides of each word. The forward and backward hidden states are concatenated to form a unified context representation of shape (batch, 512), which is passed to the decoder as the initial state.

The Decoder is a unidirectional LSTM that generates the summary one token at a time. At each step it receives the previously generated token, the current hidden state, and an attention context vector as input. Its hidden dimension is set to 512 (LATENT_DIM × 2) to match the concatenated encoder states exactly.

The Attention Mechanism is a Bahdanau-style additive attention layer placed between the encoder and decoder. At each decoding step it computes a weighted sum over all encoder outputs, allowing the decoder to focus on the most relevant parts of the source document rather than relying solely on the final encoder state. This is critical for long Arabic documents where important information may appear anywhere in the 400-token input sequence.

A Shared Embedding layer is used across both the encoder and decoder. Since both the source documents and target summaries are in Arabic, using a single embedding table ensures that identical words have identical representations on both sides of the model, which directly improves attention alignment.

**Data Preparation**
We applied teacher forcing during training, which is a standard technique for seq2seq models where the decoder is fed the ground truth token at each step rather than its own prediction. To implement this, the decoder input was constructed by dropping the last token of each summary sequence, while the decoder target was constructed by dropping the first token (sostok). This shift means the model learns to predict the next correct word given all previous correct words, which stabilizes training significantly compared to feeding predicted tokens during early epochs.

**Training**
The model was trained using the Adam optimizer with a learning rate of 3×10⁻⁴, selected after observing that the default 1×10⁻³ caused the model to plateau prematurely. We used sparse categorical cross entropy as the loss function with padding tokens (index 0) masked out to prevent the model from optimizing toward predicting zeros. Gradient clipping at a threshold of 5.0 was applied at every step to prevent exploding gradients, which are common in deep LSTM architectures. A ReduceLROnPlateau scheduler was configured to halve the learning rate when validation loss failed to improve for 2 consecutive epochs, allowing the model to escape local minima without manual intervention. Early stopping with a patience of 5 epochs was applied to restore the best weights automatically and prevent overfitting.

**Inference Model**
Because the training model relies on teacher forcing and cannot generate text autoregressively, a separate inference procedure was implemented. At inference time the encoder runs once over the full input document to produce the encoder outputs and initial hidden states. The decoder then runs in a loop, receiving a single token at each step and producing the next predicted token along with updated hidden states. The loop terminates when the model generates the eostok token or the summary reaches the maximum length of 49 tokens. This procedure is implemented in the decode_sequence function which is used directly by the evaluation step.

**Evaluation**
We evaluated the model using ROUGE scores, which is the standard metric for summarization tasks. ROUGE measures the overlap between the generated summary and the reference summary across three variants: ROUGE-1 measures unigram overlap, ROUGE-2 measures bigram overlap, and ROUGE-L measures the longest common subsequence. Evaluation was performed on 20 test samples to keep inference time practical given the sequential nature of autoregressive decoding. In addition to quantitative scores, a qualitative sample was printed showing the original document, the reference summary, and the model-generated summary side by side to allow direct visual inspection of output quality.
