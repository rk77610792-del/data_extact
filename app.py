import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile
import os

st.set_page_config(page_title="CET Chatbot", layout="wide")

st.title("🤖 CET Ranking Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload Ranking PDF", type=["pdf"])

if uploaded_file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded!")

    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings()

            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore

    query = st.text_input("Ask something (e.g. extract college data)")

    if query:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature":0, "max_length":512},
            huggingfacehub_api_token=st.secrets["HF_TOKEN"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever()
        )

        response = qa.run(query)

        st.subheader("📌 Answer")
        st.write(response)
