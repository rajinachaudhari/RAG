# RAG (Retrieval-Augmented Generation) Pipeline

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline, which is a system that retrieves relevant documents from a database based on a user query and uses them to generate more accurate responses. This README explains each component of the pipeline in beginner-friendly language.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Step-by-Step Pipeline Explanation](#step-by-step-pipeline-explanation)
6. [Functions Explained](#functions-explained)
7. [How to Run](#how-to-run)

---

## What is RAG?

Think of RAG like a librarian helping you find answers:
- **Retrieval**: The librarian searches through thousands of books to find the most relevant ones to your question
- **Augmented**: They gather these relevant documents to help formulate a better answer
- **Generation**: They use these documents to craft an informed response

In this pipeline:
- We convert documents into small, digestible pieces (chunks)
- We convert these chunks into numerical representations (embeddings)
- When you ask a question, we find the most similar chunks to your question
- These chunks can then be used by a language model to generate accurate answers

---

## Pipeline Architecture

```
Raw Data (CSV files)
    ↓
[Data Preprocessing] - Convert and clean the data
    ↓
Processed Documents (Text files)
    ↓
[Chunking] - Break documents into smaller pieces
    ↓
Chunks (with metadata)
    ↓
[Embedding] - Convert chunks to numerical vectors
    ↓
Embeddings (Vector representations)
    ↓
[Semantic Search] - Find similar chunks to user query
    ↓
Top-K Relevant Results
```

---

## Project Structure

```
RAG/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                           # Original data files
│   │   ├── multi_answers.csv         # Source data with answers
│   │   ├── multi_questions.csv       # Questions data
│   │   └── single_qna.csv            # Single Q&A pairs
│   └── processed/                     # Processed/output data
│       ├── documents.txt              # Full extracted documents
│       ├── documents_sampled_clean.txt # Cleaned sample (5%)
│       ├── chunks_df.csv             # Chunked documents table
│       └── embeddings.npy            # Embedded vectors (binary)
└── src/
    ├── data_preprocessing/
    │   ├── csv_to_documents.py       # Step 1: Extract text from CSV
    │   └── sample_dataset.py         # Step 2: Sample and clean data
    ├── chunking.py                    # Step 3: Create text chunks
    ├── embedding.py                   # Step 4: Create embeddings
    ├── query.py                       # Step 5: Run queries
    └── semantic_search.py             # Core search functionality
```

---

## Prerequisites

### Required Packages

```
numpy>=1.26          # Numerical computing library
pandas>=2.0          # Data manipulation library
scikit-learn>=1.3    # Machine learning tools (for similarity calculation)
sentence-transformers>=2.2  # Pre-trained models for embeddings
torch>=2.0           # Deep learning framework (required by sentence-transformers)
tf-keras>=           # TensorFlow (for model support)
```

### Installation

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Pipeline Explanation

### **Stage 1: Data Preprocessing** 🔄

This stage converts raw CSV data into clean, readable text documents.

#### Step 1a: CSV to Documents (`data_preprocessing/csv_to_documents.py`)

**What it does:**
- Reads the multi_answers.csv file (which contains thousands of question-answer pairs)
- Extracts the "AnswerText" column from each row
- Writes each answer as a separate document into a text file

**Why?**
- CSV files are structured data; we need unstructured text for document processing
- We're building a knowledge base of answers that can be searched

**Output:** `data/processed/documents.txt`

```python
def csv_to_documents():
    # Opens the CSV file
    # Iterates through rows in chunks (50,000 at a time for memory efficiency)
    # Extracts the answer text
    # Writes to output file
```

**Example:**
```
Input CSV Row: | ID=1 | Question="How to restart?" | AnswerText="Press the power button..." |
Output Text File: 
Press the power button...

```

#### Step 1b: Sample and Clean (`data_preprocessing/sample_dataset.py`)

**What it does:**
- Takes only 5% random sample from documents.txt (for faster processing and testing)
- Cleans the text by removing extra whitespace
- Removes empty entries

**Why?**
- Processing all data at once would be slow and memory-intensive
- Sampling is good for testing and development
- Cleaning removes noise that could affect embeddings

**Parameters:**
- `SAMPLE_RATIO = 0.05` → Takes 5% of the data randomly

**Output:** `data/processed/documents_sampled_clean.txt`

---

### **Stage 2: Chunking** ✂️

Documents are often long. We break them into smaller pieces (chunks) that are easier to search.

#### Function: `chunk_text()` (in `chunking.py`)

**What it does:**
- Splits a large text into smaller overlapping chunks
- Each chunk contains a specific number of words
- Chunks overlap to maintain context

**Parameters:**
- `text` (str): The input text to chunk
- `chunk_size` (int): Number of words per chunk (default: 40 words)
- `overlap` (int): Number of overlapping words between consecutive chunks (default: 10 words)

**Why overlapping?**
Imagine a sentence split across two chunks. Without overlap, you'd miss context:
```
Without Overlap:
Chunk 1: "The customer said they had a problem with..."
Chunk 2: "...the payment system." ← Lost context!

With Overlap (last 10 words of Chunk 1 repeat in Chunk 2):
Chunk 1: "...they had a problem with..."
Chunk 2: "...they had a problem with the payment system." ← Context preserved!
```

**Algorithm:**
```
1. Split text into words
2. Calculate step size = chunk_size - overlap
3. Iterate through words with the step size
4. Extract 'chunk_size' words at each position
5. Join words back into strings
6. Return list of chunks
```

**Example:**
```
Input: "The customer had a problem with the payment system today"
chunk_size=6, overlap=2

Chunk 1: "The customer had a problem with"
Chunk 2: "a problem with the payment system"
Chunk 3: "the payment system today"
```

**Pipeline in `chunking.py`:**

```python
# Load cleaned document
document_text = read from "documents_sampled_clean.txt"

# Create chunks
chunks = chunk_text(document_text, chunk_size=40, overlap=10)

# Create DataFrame with metadata
for each chunk:
    - domain: "customer_support" (category/source)
    - text: the chunk content

# Save as CSV for next stage
chunks_df.to_csv("data/processed/chunks_df.csv")
```

**Output:** `data/processed/chunks_df.csv`

---

### **Stage 3: Embedding** 🔢

Embeddings convert text into numerical vectors. Think of it like creating a "fingerprint" for each chunk that captures its meaning.

#### Model Used: `paraphrase-MiniLM-L3-v2`

This is a lightweight, pre-trained model that:
- Converts text sentences into 384-dimensional vectors
- Is optimized for detecting similar meanings (paraphrases)
- Runs fast without requiring a GPU

**What embeddings are:**

Imagine if you could represent the "meaning" of text as a list of numbers:
```
Chunk 1: "How to restart the system?" 
→ [0.12, -0.45, 0.89, ..., -0.23] (384 numbers)

Chunk 2: "Steps to reboot the device?"
→ [0.14, -0.43, 0.87, ..., -0.25] (384 numbers)

Notice: These numbers are very similar! The texts mean similar things.
```

#### Pipeline in `embedding.py`:

```python
# Load chunks
chunks_df = read from "chunks_df.csv"

# Initialize model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Create embeddings for all chunks
embeddings = model.encode(
    all chunk texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

# Save embeddings as binary file (efficient storage)
np.save("embeddings.npy", embeddings)
```

**Why binary format (.npy)?**
- Much smaller file size than text
- Faster to load for computations
- Preserves numerical precision

**Output:** `data/processed/embeddings.npy`

---

### **Stage 4: Semantic Search** 🔍

Given a user query, find the most similar chunks using cosine similarity.

#### Function: `semantic_search()` (in `semantic_search.py`)

**What it does:**
- Takes a user's query
- Converts the query to an embedding (same model as used for chunks)
- Calculates similarity between query embedding and all chunk embeddings
- Returns the top-k most similar chunks

**Parameters:**
- `query` (str): User's question or search term
- `model` (SentenceTransformer): The embedding model to use
- `embeddings` (np.ndarray): All pre-computed chunk embeddings
- `data` (pd.DataFrame): Chunk metadata and texts
- `top_k` (int): Number of results to return (default: 3)

#### Cosine Similarity Explained:

Imagine two vectors as arrows in space:
- Vectors pointing in similar directions = High similarity (close to 1)
- Vectors pointing in opposite directions = Low similarity (close to -1)
- Perpendicular vectors = No similarity (0)

```
           [Query Embedding]
                  ↗
           ╱─────────╲
          ╱           ╲
    [Chunk 1]   [Chunk 2]   [Chunk 3]
    High Sim    Low Sim      Very Low
    (0.92)      (0.45)       (0.12)
```

**Algorithm:**
```
1. Convert query to embedding
2. For each chunk embedding:
   - Calculate cosine similarity with query
3. Attach similarity scores to results
4. Sort by score (highest first)
5. Return top-k results
```

#### Pipeline in `query.py`:

```python
# Load all preprocessed data
chunks_df = read from "chunks_df.csv"
embeddings = load from "embeddings.npy"
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# User query
query = "Can you get the extra wires needed on amazon?"

# Search
results = semantic_search(
    query=query,
    model=model,
    embeddings=embeddings,
    data=chunks_df,
    top_k=3  # Get top 3 results
)

# Display
Print domain, text, and similarity score for each result
```

**Example Output:**
```
Query: "How to restart?"

Top matching chunks:
┌─────────────┬─────────────────────────────┬───────┐
│ domain      │ text                        │ score │
├─────────────┼─────────────────────────────┼───────┤
│ cust_supp   │ To restart press power...   │ 0.923 │
│ cust_supp   │ Reboot system using menu... │ 0.891 │
│ cust_supp   │ How to shut down properly..│ 0.756 │
└─────────────┴─────────────────────────────┴───────┘
```

---

## Functions Explained

### `chunk_text(text, chunk_size=40, overlap=10)` → `List[str]`

**Location:** [src/chunking.py](src/chunking.py)

**Input:**
- `text`: A large text string to split into chunks

**Output:**
- List of text chunks (strings)

**What it does step-by-step:**
1. Splits the entire text into individual words
2. Creates overlapping chunks based on word count
3. Returns list of chunks

**Code Walkthrough:**
```python
words = text.split()  # ["The", "dog", "ran", "fast"]

step = chunk_size - overlap  # = 40 - 10 = 30

# Iterate with step size
for i in range(0, len(words), step):
    chunk = " ".join(words[i : i + chunk_size])
    # Get 40 words starting from position i
    chunks.append(chunk)
```

---

### `csv_to_documents()`

**Location:** [src/data_preprocessing/csv_to_documents.py](src/data_preprocessing/csv_to_documents.py)

**What it does:**
- Reads CSV file in chunks (to handle large files efficiently)
- Extracts answer text from each row
- Filters out empty/invalid entries
- Writes to output text file

**Key Features:**
- Uses `chunksize=50_000` to avoid loading entire file in memory
- Handles different data types (converts to string)
- Skips "nan" values

---

### `semantic_search(query, model, embeddings, data, top_k=3)` → `pd.DataFrame`

**Location:** [src/semantic_search.py](src/semantic_search.py)

**What it does:**
1. Encodes the query into an embedding
2. Calculates cosine similarity with all chunk embeddings
3. Attaches scores to results
4. Sorts and returns top-k results

**Returns:**
A DataFrame with columns:
- `domain`: Source category
- `text`: Chunk content
- `score`: Similarity score (0-1)

---

## How to Run

### **Complete Pipeline Execution:**

#### Step 1: Preprocess Data
```bash
python src/data_preprocessing/csv_to_documents.py
python src/data_preprocessing/sample_dataset.py
```

**Output:** `documents_sampled_clean.txt`

#### Step 2: Create Chunks
```bash
python src/chunking.py
```

**Output:** `chunks_df.csv`, logs total chunks created

#### Step 3: Generate Embeddings
```bash
python src/embedding.py
```

**Output:** `embeddings.npy`, logs embedding shape

#### Step 4: Query and Search
```bash
python src/query.py
```

**Output:** Top-3 similar chunks with similarity scores

### **Modify Queries:**

Edit the query in [src/query.py](src/query.py#L16):
```python
query = "Your custom question here?"
```

Then run:
```bash
python src/query.py
```

---

## Understanding the Data Flow

```
documents_sampled_clean.txt (Raw text)
        ↓
    [CHUNKING]
        ↓
chunks_df.csv (Chunks + metadata)
        ↓
    [EMBEDDING]
        ↓
embeddings.npy (Numerical vectors)
        ↓
    [QUERY INPUT]
        ↓
    [SEMANTIC SEARCH]
        ↓
Top-K Results (Most similar chunks)
```

---

## Key Concepts for Beginners

### Vectorization
Converting text into numbers that computers can understand and compare mathematically.

### Similarity Metric
A way to measure how similar two pieces of text are based on their embeddings.

### Top-K Results
Instead of getting one result, we get the K (e.g., 3) best results, giving users options.

### Pre-trained Models
Models already trained on billions of text examples. We use them instead of training from scratch (which is slow and requires lots of data).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` | Ensure you've run preprocessing steps first |
| `CUDA out of memory` | The model will automatically use CPU instead |
| `Slow embedding generation` | This is normal for large datasets; use the sampled version for testing |
| `Low similarity scores` | Your query might be too different from the documents; try different keywords |

---

## Next Steps (Future Enhancements)

- [ ] Change into heavy models and increase datasets for better accuracy (e.g., `all-MiniLM-L6-v2`)
- [ ] Integrate with an LLM (like GPT) to generate responses using retrieved chunks
- [ ] Implement caching for faster queries
- [ ] Add support for other languages
- [ ] Create a REST API for production deployment

---

## Dependencies Explained

- **NumPy**: Fast numerical operations on arrays
- **Pandas**: Table/DataFrame manipulation (like Excel in Python)
- **Scikit-learn**: Machine learning tools, including similarity calculations
- **Sentence-Transformers**: Pre-trained models for converting text to embeddings
- **PyTorch**: Deep learning framework (required by Sentence-Transformers)

---

## License

See [LICENSE](LICENSE) file for details.
