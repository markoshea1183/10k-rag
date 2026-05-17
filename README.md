# 10-K RAG Financial Research Assistant

This project builds a Retrieval-Augmented Generation (RAG) system for analyzing company 10-K filings. The application allows users to ask natural-language financial research questions and retrieve relevant filing passages using semantic vector search. Retrieved filing context is then used to generate source-grounded answers.

The system combines:
- PDF parsing and text extraction
- Text chunking
- Semantic embeddings
- ChromaDB vector search
- Metadata filtering
- OpenAI-powered answer generation
- Streamlit deployment

The dataset currently includes 2024 and 2025 10-K filings from:
- NVIDIA
- Apple
- Amazon

Example questions:
- “What risks does Nvidia identify related to export controls and China?”
- “How does Amazon discuss logistics and fulfillment risks?”
- “What regulatory risks does Apple discuss?”
- “What competitive threats does Nvidia mention?”

## Project Structure

```text
10k-rag/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── chroma_db/
├── notebooks/
│   └── 10k_rag_final.ipynb
├── requirements.txt
├── .env
└── README.md
```

## Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- OpenAI API
- PyTorch
- Transformers
- LangChain utilities

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd 10k-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create OpenAI API key file

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

## Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/10k_rag_final.ipynb
```

## Running the Streamlit App

From the project root:

```bash
streamlit run app/streamlit_app.py
```

If using a (my) specific Python installation:

```bash
/usr/local/bin/python3.10 -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Main Features

- Semantic search over company filings
- Company and year filtering
- Vector similarity retrieval using embeddings
- Source-grounded answer generation
- Interactive Streamlit interface
- Persistent local vector database

## Educational Purpose

This project was developed as a graduate-level machine learning and NLP project demonstrating practical applications of Retrieval-Augmented Generation (RAG) systems for financial document analysis.