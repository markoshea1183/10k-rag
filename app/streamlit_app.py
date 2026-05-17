from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

st.set_page_config(page_title="10-K RAG Assistant", layout="wide")

st.title("10-K RAG Financial Research Assistant")
st.write("Retrieval-only demo: semantic search over company 10-K filings.")

st.write("Chroma path:", str(CHROMA_PATH))
st.write("Chroma exists:", CHROMA_PATH.exists())


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name="company_10k_filings")


model = load_embedding_model()
collection = load_collection()

company = st.sidebar.selectbox("Company", ["All", "nvidia", "amazon", "apple"])
year = st.sidebar.selectbox("Year", ["All", "2024", "2025"])
n_results = st.sidebar.slider("Number of sources", 1, 10, 5)

question = st.text_input(
    "Ask a question:",
    "What risks does Nvidia identify related to export controls and China?"
)

if st.button("Retrieve Sources"):
    query_embedding = model.encode([question]).tolist()

    filters = []
    if company != "All":
        filters.append({"company": company})
    if year != "All":
        filters.append({"year": year})

    where_filter = None
    if len(filters) == 1:
        where_filter = filters[0]
    elif len(filters) > 1:
        where_filter = {"$and": filters}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where_filter
    )

    st.subheader("Retrieved Sources")

    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]

        with st.expander(
            f"Source {i}: {meta['company']} {meta['year']} | "
            f"{meta['source_file']} | page {meta['page']}"
        ):
            st.write(doc[:1500])