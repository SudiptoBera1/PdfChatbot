import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import tempfile
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

st.title("📄 Chat with your PDF")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf.read())
        pdf_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever()
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask something from the PDF:")

    if question:
        result = qa({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        answer = result["answer"]

        st.write("**You:**", question)
        st.write("**Bot:**", answer)

        st.session_state.chat_history.append((question, answer))
