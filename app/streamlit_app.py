from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

load_dotenv()

@st.cache_resource
def load_openai_client():
    return OpenAI()

openai_client = load_openai_client()

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="10-K RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("10-K RAG Financial Research Assistant")

st.write(
    "Ask questions about company 10-K filings. "
    "The app retrieves relevant filing passages using vector search "
    "and uses OpenAI's API to generate a source-grounded answer."
)


# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name="company_10k_filings")


model = load_embedding_model()
collection = load_collection()


# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text):
    return " ".join(text.split())


def build_where_filter(company, year):
    filters = []

    if company != "All":
        filters.append({"company": company})

    if year != "All":
        filters.append({"year": year})

    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}


def retrieve_sources(question, company, year, n_results):
    query_embedding = model.encode([question]).tolist()

    if company != "All":
        where_filter = build_where_filter(company, year)

        return collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter
        )

    # Balanced retrieval across all companies
    companies = ["nvidia", "amazon", "apple"]
    combined_docs = []
    combined_metas = []
    combined_distances = []

    per_company_k = max(1, n_results // len(companies) + 1)

    for c in companies:
        where_filter = build_where_filter(c, year)

        res = collection.query(
            query_embeddings=query_embedding,
            n_results=per_company_k,
            where=where_filter
        )

        combined_docs.extend(res["documents"][0])
        combined_metas.extend(res["metadatas"][0])
        combined_distances.extend(res["distances"][0])

    ranked = sorted(
        zip(combined_docs, combined_metas, combined_distances),
        key=lambda x: x[2]
    )[:n_results]

    docs, metas, distances = zip(*ranked)

    return {
        "documents": [list(docs)],
        "metadatas": [list(metas)],
        "distances": [list(distances)]
    }


def format_context(results, max_chars_per_source=700):
    context_blocks = []

    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]

        source_header = (
            f"[Source {i}: {meta['company']} {meta['year']}, "
            f"{meta['source_file']}, page {meta['page']}]"
        )

        context_blocks.append(
            source_header + "\n" + clean_text(doc)[:max_chars_per_source]
        )

    return "\n\n".join(context_blocks)


def generate_answer(question, context):
    prompt = f"""
You are a financial research assistant.

Use only the retrieved 10-K context below to answer the question.
Answer in 3-5 concise sentences.
Cite sources using [Source 1], [Source 2], etc.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question:
{question}
"""

    st.write("Using OpenAI API for generation")

    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.2,
        max_output_tokens=350
    )

    return response.output_text


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Search Controls")

    company = st.selectbox(
        "Company",
        ["All", "nvidia", "amazon", "apple"]
    )

    year = st.selectbox(
        "Year",
        ["All", "2024", "2025"]
    )

    n_results = st.slider(
        "Number of retrieved sources",
        min_value=1,
        max_value=10,
        value=5
    )

    max_chars = st.slider(
        "Characters shown per source",
        min_value=300,
        max_value=1500,
        value=1000,
        step=100
    )

    use_generation = st.checkbox(
        "Generate answer with OpenAI",
        value=True
    )


# -----------------------------
# Example prompts
# -----------------------------
example_questions = {
    "nvidia": "What risks does Nvidia identify related to export controls and China?",
    "amazon": "How does Amazon discuss fulfillment and logistics risks?",
    "apple": "What supply chain risks does Apple discuss?",
    "All": "What macroeconomic risks are discussed across the filings?"
}

default_question = example_questions.get(company, example_questions["All"])

question = st.text_input(
    "Ask a question:",
    value=default_question
)


# -----------------------------
# Main app logic
# -----------------------------
if st.button("Run RAG Query"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant 10-K passages..."):
            results = retrieve_sources(
                question=question,
                company=company,
                year=year,
                n_results=n_results
            )

        context = format_context(
            results,
            max_chars_per_source=700
        )

        if use_generation:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, context)

            st.subheader("Generated Answer")
            st.write(answer)
        else:
            st.subheader("Generated Answer")
            st.info("Generation disabled. Showing retrieved sources only.")

        st.subheader("Retrieved Sources")

        for i, doc in enumerate(results["documents"][0], 1):
            meta = results["metadatas"][0][i - 1]
            displayed_text = clean_text(doc)[:max_chars]

            with st.expander(
                f"Source {i}: {meta['company']} {meta['year']} | "
                f"{meta['source_file']} | page {meta['page']}"
            ):
                st.write(displayed_text)