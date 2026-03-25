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
