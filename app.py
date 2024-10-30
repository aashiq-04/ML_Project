import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
import time

st.title("ChatGroq")

web_page = st.text_input(label="Enter Web Page URL to continue")
prompt = st.text_input("Input your prompt here")

groq_api_key = os.environ['GROQ_API_KEY']

# Initialize session state for vector store
if web_page and "vectors" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.loader = WebBaseLoader(web_page)
    
    try:
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Web page processed successfully!")
    except Exception as e:
        st.error(f"Error processing the web page: {str(e)}")
        st.stop()

# Only create the retriever and chains if vectors are available
if "vectors" in st.session_state:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        print("Response Time:", time.process_time() - start)
        st.write(response['answer'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.info("Please enter a valid web page URL to process before asking questions.")