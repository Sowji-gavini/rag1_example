import streamlit as st

from utils import (
    load_pdf,
    chunk_text,
    get_embedding_model,
    create_vectorstore,
    retrieve_chunks,
    ask_groq,
)

# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered",
)

# ===================================================
# TITLE
# ===================================================

st.title("RAG CHATBOT")

st.write(
    "Upload a PDF and ask questions"
)

# ===================================================
# FILE UPLOAD
# ===================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

# ===================================================
# PROCESS PDF
# ===================================================

if uploaded_file:

    try:

        # -----------------------------------
        # LOAD PDF
        # -----------------------------------

        with st.spinner(
            "Reading PDF..."
        ):

            text = load_pdf(
                uploaded_file
            )

        st.success(
            "PDF LOADED"
        )

        # -----------------------------------
        # CHUNKING
        # -----------------------------------

        chunks = chunk_text(text)

        st.success(
            f"{len(chunks)} CHUNKS CREATED"
        )

        # -----------------------------------
        # LOAD MODEL
        # -----------------------------------

        with st.spinner(
            "Loading embedding model..."
        ):

            model = get_embedding_model()

        st.success(
            "MODEL LOADED"
        )

        # -----------------------------------
        # VECTOR STORE
        # -----------------------------------

        with st.spinner(
            "Creating vector store..."
        ):

            index = create_vectorstore(
                chunks,
                model,
            )

        st.success(
            "VECTOR STORE CREATED"
        )

        # ===================================================
        # CHATBOT
        # ===================================================

        st.divider()

        st.subheader(
            "CHAT WITH PDF"
        )

        question = st.text_input(
            "Ask a question from the PDF"
        )

        # -----------------------------------
        # QUESTION
        # -----------------------------------

        if question:

            with st.spinner(
                "Thinking..."
            ):

                # Retrieve relevant chunks
                retrieved_chunks = (
                    retrieve_chunks(
                        question,
                        chunks,
                        model,
                        index,
                        top_k=4,
                    )
                )

                # Ask Groq
                answer = ask_groq(
                    question,
                    retrieved_chunks,
                )

            # -----------------------------------
            # SHOW ANSWER
            # -----------------------------------

            st.subheader(
                "ANSWER"
            )

            st.write(answer)

            # -----------------------------------
            # SHOW CHUNKS
            # -----------------------------------

            with st.expander(
                "Retrieved Chunks"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1,
                ):

                    st.write(
                        f"Chunk {i}"
                    )

                    st.info(chunk)

    except Exception as e:

        st.error(str(e))