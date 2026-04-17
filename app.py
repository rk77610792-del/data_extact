import streamlit as st
import tempfile
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Page config
st.set_page_config(page_title="CET Chatbot", layout="wide")

st.title("🤖 CET Ranking Chatbot (No Errors Version)")

# Load embedding model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Split text into chunks
def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload Ranking PDF", type=["pdf"])

if uploaded_file:

    st.success("✅ PDF uploaded successfully!")

    # Extract text
    text = extract_text_from_pdf(uploaded_file)

    # Split into chunks
    chunks = split_text(text)

    # Convert to embeddings
    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success("✅ PDF processed!")

    # HuggingFace client
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=st.secrets["HF_TOKEN"]
    )

    # User query
    query = st.text_input("💬 Ask your question")

    if query:
        # Embed query
        query_vec = model.encode([query])

        # Search similar chunks
        D, I = index.search(np.array(query_vec), k=5)

        context = "\n".join([chunks[i] for i in I[0]])

        # Prompt
        prompt = f"""
You are an expert data extractor.

From the given context, extract:
CODE, COLLEGE NAME, COURSE CODE, COURSE NAME, CET NO, LOCATION

Return in clean structured format.

Context:
{context}

Question:
{query}
"""

        with st.spinner("🤖 Generating answer..."):
            response = client.text_generation(
                prompt,
                max_new_tokens=500,
                temperature=0
            )

        st.subheader("📌 Answer")
        st.write(response)

# Reset button
if st.button("🔄 Reset"):
    st.experimental_rerun()
