import streamlit as st

st.set_page_config(page_title="RAG")

st.title("RAG CHATBOT")

st.write("APP STARTED")

try:

    from utils import (
        load_pdf,
        chunk_text,
        get_embeddings,
        init_qdrant_collection,
        upload_to_qdrant,
        retrieve_chunks,
        ask_groq,
    )

    st.success("UTILS IMPORTED")

except Exception as e:

    st.error(f"IMPORT ERROR: {str(e)}")

    st.stop()

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

if uploaded_file:

    try:

        st.write("READING PDF")

        text = load_pdf(uploaded_file)

        st.success("PDF LOADED")

        chunks = chunk_text(text)

        st.success(f"{len(chunks)} CHUNKS")

        embeddings = get_embeddings(chunks[:2])

        st.success("EMBEDDINGS WORKING")

    except Exception as e:

        st.error(str(e))
