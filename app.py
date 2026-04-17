import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="CET Chatbot", layout="wide")
st.title("🤖 CET Ranking Chatbot")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- FUNCTIONS ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📄 Upload Ranking PDF", type=["pdf"])

if uploaded_file:

    st.success("✅ PDF uploaded successfully!")

    # Extract text
    text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("❌ Could not extract text (PDF may be scanned)")
        st.stop()

    # Split text
    chunks = split_text(text)

    # Embeddings
    embeddings = model.encode(chunks)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success("✅ PDF processed successfully!")

    # HuggingFace Client
    client = InferenceClient(
        token=st.secrets["HF_TOKEN"]
    )

    # ---------------- CHAT INPUT ----------------
    query = st.text_input(
        "💬 Ask your question",
        placeholder="Extract CODE, COLLEGE NAME, COURSE CODE, COURSE NAME, CET NO, LOCATION"
    )

    if query:
        # Embed query
        query_vec = model.encode([query])

        # Search top chunks
        D, I = index.search(np.array(query_vec), k=5)
        context = "\n".join([chunks[i] for i in I[0]])

        # Prompt
        prompt = f"""
You are an expert data extractor.

Extract the following fields:
- CODE
- COLLEGE NAME
- COURSE CODE
- COURSE NAME
- CET NO
- LOCATION

Return output in clean structured format (table or CSV).

Context:
{context}

User Question:
{query}
"""

        # ---------------- CHAT MODEL ----------------
        with st.spinner("🤖 Generating answer..."):
            response = client.chat.completions.create(
                model="HuggingFaceH4/zephyr-7b-beta",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )

            answer = response.choices[0].message.content

        # ---------------- OUTPUT ----------------
        st.subheader("📌 Answer")
        st.write(answer)

# ---------------- RESET ----------------
if st.button("🔄 Reset"):
    st.rerun()
