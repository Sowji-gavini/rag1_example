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
# PAGE
# ===================================================

st.set_page_config(
    page_title="RAG Chatbot"
)

st.title("RAG CHATBOT")

st.write(
    "Upload PDF and ask questions"
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

        # -------------------------
        # LOAD PDF
        # -------------------------

        text = load_pdf(
            uploaded_file
        )

        st.success("PDF LOADED")

        # -------------------------
        # CHUNKS
        # -------------------------

        chunks = chunk_text(text)

        st.success(
            f"{len(chunks)} CHUNKS CREATED"
        )

        # -------------------------
        # MODEL
        # -------------------------

        with st.spinner(
            "LOADING MODEL..."
        ):

            model = get_embedding_model()

        st.success(
            "MODEL LOADED"
        )

        # -------------------------
        # VECTORSTORE
        # -------------------------

        with st.spinner(
            "CREATING VECTORSTORE..."
        ):

            index, embeddings = (
                create_vectorstore(
                    chunks,
                    model,
                )
            )

        st.success(
            "VECTORSTORE CREATED"
        )

        # ===================================================
        # CHAT
        # ===================================================

        st.divider()

        question = st.text_input(
            "Ask Question"
        )

        if question:

            with st.spinner(
                "THINKING..."
            ):

                retrieved_chunks = (
                    retrieve_chunks(
                        question,
                        chunks,
                        model,
                        index,
                    )
                )

                answer = ask_groq(
                    question,
                    retrieved_chunks,
                )

            st.subheader(
                "ANSWER"
            )

            st.write(answer)

            with st.expander(
                "Retrieved Chunks"
            ):

                for chunk in retrieved_chunks:

                    st.info(chunk)

    except Exception as e:

        st.error(str(e))