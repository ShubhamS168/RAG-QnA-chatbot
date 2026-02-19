import streamlit as st
from io import BytesIO

from utils import (
    read_pdf,
    read_txt,
    read_docx,
    read_csv,
    clean_text,
    chunk_text,
)
from retriever import Retriever
from groq_qa import GroqQA   # 🔁 switched from GeminiQA
from config import Config


# ===============================
# Cache heavy components
# ===============================
@st.cache_resource
def get_retriever():
    return Retriever()


@st.cache_resource
def get_qa_engine():
    return GroqQA()


retriever = get_retriever()
qa_engine = get_qa_engine()


# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="RAG Q&A Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Q&A Chatbot")
st.markdown(
    """
Upload documents (PDF, TXT, DOCX, CSV), ask questions,  
and get answers powered by **Retrieval-Augmented Generation (RAG)**.
"""
)


# ===============================
# Session state
# ===============================
st.session_state.setdefault("faiss_index", None)
st.session_state.setdefault("document_texts", [])
st.session_state.setdefault("processed_chunks", [])
st.session_state.setdefault("uploaded_file_names", [])


# ===============================
# Sidebar – Upload documents
# ===============================
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=Config.SUPPORTED_FILE_TYPES,
    accept_multiple_files=True,
)

if uploaded_files:
    new_files = False

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.uploaded_file_names:
            continue

        new_files = True
        st.session_state.uploaded_file_names.append(uploaded_file.name)

        ext = uploaded_file.name.split(".")[-1].lower()
        raw_text = ""

        try:
            file_content = BytesIO(uploaded_file.getvalue())

            if ext == "pdf":
                raw_text = read_pdf(file_content)
            elif ext == "txt":
                raw_text = read_txt(file_content)
            elif ext == "docx":
                raw_text = read_docx(file_content)
            elif ext == "csv":
                raw_text = read_csv(file_content)

            if raw_text.strip():
                st.session_state.document_texts.append(raw_text)
                st.sidebar.success(f"Processed {uploaded_file.name}")
            else:
                st.sidebar.warning(f"No text found in {uploaded_file.name}")

        except Exception as e:
            st.sidebar.error(f"Failed to process {uploaded_file.name}: {e}")

    if new_files and st.session_state.document_texts:
        with st.spinner("Processing documents and building vector index..."):
            try:
                combined_text = "\n\n".join(st.session_state.document_texts)
                cleaned_text = clean_text(combined_text)

                chunks = chunk_text(
                    cleaned_text,
                    Config.CHUNK_SIZE,
                    Config.CHUNK_OVERLAP,
                )

                st.session_state.processed_chunks = chunks
                st.session_state.faiss_index, _ = retriever.build_faiss_index(chunks)

                st.sidebar.info(f"Indexed {len(chunks)} text chunks")

            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")
                st.session_state.faiss_index = None
                st.session_state.processed_chunks = []


# ===============================
# Sidebar – Uploaded files list
# ===============================
if st.session_state.uploaded_file_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Uploaded Files")
    for name in st.session_state.uploaded_file_names:
        st.sidebar.markdown(f"- {name}")


# ===============================
# Q&A Section
# ===============================
st.header("❓ Ask a Question")

if st.session_state.faiss_index is None:
    st.info("Upload documents to enable question answering.")
else:
    question = st.text_input("Enter your question")

    show_context = st.checkbox("Show Retrieved Context")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    contexts = retriever.retrieve_context(
                        # query=question,
                        # faiss_index=st.session_state.faiss_index,
                        # text_chunks=st.session_state.processed_chunks,
                        # k=Config.TOP_K_RETRIEVAL,
                        question,
                        st.session_state.faiss_index,
                        st.session_state.processed_chunks,
                        Config.TOP_K_RETRIEVAL,
                    )

                    if show_context:
                        st.subheader("Retrieved Context")
                        if contexts:
                            for i, ctx in enumerate(contexts, 1):
                                st.markdown(f"**Context {i}:**")
                                st.info(ctx)
                        else:
                            st.warning("No relevant context found.")

                    answer = qa_engine.generate_answer(question, contexts)

                    st.subheader("AI Answer")
                    st.success(answer)

                except Exception as e:
                    st.error(f"Q&A failed: {e}")


# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    """
**How it works**
1. Documents are split into chunks  
2. FAISS retrieves the most relevant chunks  
3. The LLM answers **only using retrieved context**

This is a true **RAG pipeline**, not a chatbot with memory.
"""
)


# ===============================
# Reset button
# ===============================
if st.sidebar.button("Clear All Uploaded Data"):
    st.session_state.faiss_index = None
    st.session_state.document_texts = []
    st.session_state.processed_chunks = []
    st.session_state.uploaded_file_names = []
    st.rerun()






























































































# import streamlit as st
# import os
# from io import BytesIO

# from utils import read_pdf, read_txt, read_docx, read_csv, clean_text, chunk_text
# from retriever import Retriever
# from gemini_qa import GeminiQA
# from config import Config

# # --- Initialize components ---
# # Initialize Retriever and GeminiQA outside of functions to avoid re-initialization on every rerun
# # Use st.cache_resource for heavy objects like models and FAISS index
# @st.cache_resource
# def get_retriever():
#     """Caches the Retriever instance."""
#     return Retriever()

# @st.cache_resource
# def get_gemini_qa():
#     """Caches the GeminiQA instance."""
#     return GeminiQA()

# retriever = get_retriever()
# gemini_qa = get_gemini_qa()

# # --- Streamlit App Configuration ---
# st.set_page_config(
#     page_title="RAG Q&A Chatbot",
#     page_icon="📚",
#     layout="wide"
# )

# st.title("📚 RAG Q&A Chatbot with Gemini")
# st.markdown("""
#     Upload your documents (PDF, TXT, DOCX, CSV), ask questions, and get answers
#     powered by Google Gemini and document retrieval.
# """)

# # --- Session State Management ---
# if "faiss_index" not in st.session_state:
#     st.session_state.faiss_index = None
# if "document_texts" not in st.session_state:
#     st.session_state.document_texts = []
# if "processed_chunks" not in st.session_state:
#     st.session_state.processed_chunks = []
# if "uploaded_file_names" not in st.session_state:
#     st.session_state.uploaded_file_names = []

# # --- Document Upload Section ---
# st.sidebar.header("Upload Documents")
# uploaded_files = st.sidebar.file_uploader(
#     "Choose files to upload",
#     type=Config.SUPPORTED_FILE_TYPES,
#     accept_multiple_files=True
# )

# if uploaded_files:
#     new_files_uploaded = False
#     for uploaded_file in uploaded_files:
#         if uploaded_file.name not in st.session_state.uploaded_file_names:
#             new_files_uploaded = True
#             st.session_state.uploaded_file_names.append(uploaded_file.name)

#             file_extension = uploaded_file.name.split(".")[-1].lower()
#             raw_text = ""
#             try:
#                 # Read file content into BytesIO to pass to utility functions
#                 file_content = BytesIO(uploaded_file.getvalue())

#                 if file_extension == "pdf":
#                     raw_text = read_pdf(file_content)
#                 elif file_extension == "txt":
#                     raw_text = read_txt(file_content)
#                 elif file_extension == "docx":
#                     raw_text = read_docx(file_content)
#                 elif file_extension == "csv":
#                     raw_text = read_csv(file_content)
#                 else:
#                     st.sidebar.warning(f"Unsupported file type: {file_extension} for {uploaded_file.name}")
#                     continue

#                 if raw_text:
#                     st.session_state.document_texts.append(raw_text)
#                     st.sidebar.success(f"Processed {uploaded_file.name}")
#                 else:
#                     st.sidebar.error(f"Could not extract text from {uploaded_file.name}. It might be empty or corrupted.")

#             except Exception as e:
#                 st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")

#     if new_files_uploaded and st.session_state.document_texts:
#         with st.spinner("Processing documents and building index... This may take a moment."):
#             try:
#                 combined_text = "\n\n".join(st.session_state.document_texts)
#                 cleaned_text = clean_text(combined_text)
#                 chunks = chunk_text(cleaned_text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
#                 st.session_state.processed_chunks = chunks
#                 st.session_state.faiss_index, _ = retriever.build_faiss_index(chunks)
#                 st.sidebar.info(f"Indexed {len(chunks)} text chunks.")
#             except Exception as e:
#                 st.sidebar.error(f"Failed to build index: {e}")
#                 st.session_state.faiss_index = None
#                 st.session_state.processed_chunks = []

# # Display uploaded file names in sidebar
# if st.session_state.uploaded_file_names:
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("Uploaded Files:")
#     for fname in st.session_state.uploaded_file_names:
#         st.sidebar.markdown(f"- {fname}")

# # --- Q&A Section ---
# st.header("Ask a Question")

# if st.session_state.faiss_index is None:
#     st.info("Please upload documents to enable the Q&A functionality.")
# else:
#     question = st.text_input("Enter your question here:", key="question_input")

#     show_context_toggle = st.checkbox("Show Retrieved Context", value=False)

#     if st.button("Get Answer", key="get_answer_button"):
#         if question:
#             with st.spinner("Retrieving context and generating answer..."):
#                 try:
#                     # Retrieve context
#                     retrieved_contexts = retriever.retrieve_context(
#                         question,
#                         st.session_state.faiss_index,
#                         st.session_state.processed_chunks,
#                         k=Config.TOP_K_RETRIEVAL
#                     )

#                     if show_context_toggle:
#                         st.subheader("Retrieved Contexts:")
#                         if retrieved_contexts:
#                             for i, context_item in enumerate(retrieved_contexts):
#                                 st.markdown(f"**Context {i+1}:**")
#                                 st.info(context_item)
#                         else:
#                             st.warning("No relevant context found for your question.")

#                     # Generate answer
#                     answer = gemini_qa.generate_answer(question, retrieved_contexts)
#                     st.subheader("AI Answer:")
#                     st.success(answer)

#                 except Exception as e:
#                     st.error(f"An error occurred during Q&A: {e}")
#         else:
#             st.warning("Please enter a question.")

# # --- Instructions and Footer ---
# st.markdown("---")
# st.markdown("""
#     **Instructions:**
#     1.  **Upload Documents:** Use the sidebar to upload PDF, TXT, DOCX, or CSV files.
#     2.  **Ask a Question:** Once documents are processed and indexed, type your question in the input box.
#     3.  **Get Answer:** Click "Get Answer" to see the AI-generated response based on your documents.
#     4.  **Show Context:** Toggle "Show Retrieved Context" to see which parts of your documents were used.
    
#     *Note: The first time you upload documents or run the app, it might take a moment to download the SentenceTransformer model and build the FAISS index.*
# """)

# # Clear session state button (optional, for debugging/resetting)
# if st.sidebar.button("Clear All Uploaded Data"):
#     st.session_state.faiss_index = None
#     st.session_state.document_texts = []
#     st.session_state.processed_chunks = []
#     st.session_state.uploaded_file_names = []
#     st.rerun()
#     st.sidebar.success("All uploaded data cleared.")
