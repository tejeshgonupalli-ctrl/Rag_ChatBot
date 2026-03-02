import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from PyPDF2 import PdfReader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


load_dotenv()

def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def load_chat_model():
    return GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)

def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    # Return a list of Document objects (each must have page_content) because
    # FAISS.from_documents expects Document-like objects.
    return [Document(page_content=chunk) for chunk in chunks]

def create_vector_store(chunks, embeddings):
    # Accept either a list of Document objects or a list of raw strings.
    if len(chunks) == 0:
        return FAISS.from_documents([], embeddings)

    # If the items are strings, convert to Document objects.
    if isinstance(chunks[0], str):
        docs = [Document(page_content=c) for c in chunks]
    else:
        docs = chunks

    return FAISS.from_documents(docs, embeddings)

def save_vector_store(vector_store, file_name):
    vector_store.save_local(file_name)

def load_vector_store(file_name, embeddings):
    return FAISS.load_local(file_name, embeddings, allow_dangerous_deserialization=True)

def retrieve_similar_docs(vector_store, query, k=3):
    return vector_store.similarity_search(query, k=k)


st.set_page_config(page_title="RAG Chatbot with Google Gemini", page_icon=":robot_face:", layout="wide")
st.title("RAG Chatbot with Google Gemini")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.embeddings = load_embeddings_model()
    st.session_state.chat_model = load_chat_model()

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_pdf(uploaded_file)
        chunks = create_chunks(text)
        st.session_state.vector_store = create_vector_store(chunks, st.session_state.embeddings)
        save_vector_store(st.session_state.vector_store, "vector_store")
        st.success("Vector store created and saved!")

    if os.path.exists("vector_store"):
        if st.button("Load Existing Vector Store"):
            st.session_state.vector_store = load_vector_store("vector_store", st.session_state.embeddings)
            st.success("Vector store loaded!")

## Creating a chatbot interface on the main window
st.header("Chat with your PDF")
if st.session_state.vector_store is None:
    st.info("Please upload a PDF file to create or load a vector store.")
else:
    query = st.text_input("Enter your question:")
    if query and st.button("Get Answer"):
        with st.spinner("Generating response..."):
            similar_docs = retrieve_similar_docs(st.session_state.vector_store, query)
            # st.markdown("**Top 3 relevant document chunks:**")
            # for i, doc in enumerate(similar_docs, 1):
            #     st.markdown(f"**Chunk {i}:** {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
            context = "\n".join([doc.page_content for doc in similar_docs])
            prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
            response = st.session_state.chat_model.invoke(prompt)
            st.markdown(f"**Response:** {response}")
        #st.markdown(f"**Response:** {response.content}")