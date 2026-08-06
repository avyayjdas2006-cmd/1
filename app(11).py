__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


st.set_page_config(page_title="PDF QA App", page_icon="📄")
st.title(" PDF Assistant")
st.caption("Upload a PDF document and start chatting with it.")


if "GROQ_API_KEY" in st.secrets:
  api_key = st.secrets["GROQ_API_KEY"]
else:
  with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")

if not api_key:
  st.info(
      "👉 Please configure your GROQ_API_KEY in Streamlit Secrets or enter it"
      " here."
  )
  st.stop()

os.environ["GROQ_API_KEY"] = api_key


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_pdf_path = temp_file.name

    
    with st.spinner("Processing PDF document..."):
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load_and_split()

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory
    )

    st.success("PDF loaded! Ask your questions below.")

    
    user_prompt = st.chat_input("Ask something about the document...")

    if user_prompt:
        with st.spinner("Thinking..."):
            qa_chain.run(user_prompt)

    
    if "memory" in st.session_state:
        messages = st.session_state.memory.chat_memory.messages
        for message in messages:
            role = "user" if message.type == "human" else "assistant"
            with st.chat_message(role):
                st.write(message.content)