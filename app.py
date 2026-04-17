import streamlit as st
import tempfile

# Updated LangChain imports (IMPORTANT)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Page config
st.set_page_config(page_title="CET Ranking Chatbot", layout="wide")

st.title("🤖 CET Ranking Chatbot")
st.write("Upload your ranking PDF and ask for structured data extraction")

# Upload PDF
uploaded_file = st.file_uploader("Upload Ranking PDF", type=["pdf"])

if uploaded_file:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    # Process PDF only once
    if "vectorstore" not in st.session_state:
        with st.spinner("🔄 Processing PDF..."):

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(documents)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings()

            # Create vector DB
            vectorstore = FAISS.from_documents(chunks, embeddings)

            st.session_state.vectorstore = vectorstore

        st.success("✅ PDF processed successfully!")

    # User query
    query = st.text_input(
        "💬 Ask your question",
        placeholder="Example: Extract CODE, COLLEGE NAME, COURSE CODE, COURSE NAME, CET NO, LOCATION"
    )

    if query:

        with st.spinner("🤖 Generating answer..."):

            # Load LLM (HuggingFace)
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",  # fast & stable
                model_kwargs={
                    "temperature": 0,
                    "max_length": 512
                },
                huggingfacehub_api_token=st.secrets["HF_TOKEN"]
            )

            # Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever()
            )

            response = qa.run(query)

            st.subheader("📌 Answer")
            st.write(response)

    # Optional: Clear session
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.experimental_rerun()
