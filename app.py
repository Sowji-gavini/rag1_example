import streamlit as st

# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered",
)

st.title("RAG CHATBOT")

st.write("Upload a PDF and ask questions")


# ===================================================
# IMPORT UTILS
# ===================================================

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

    st.error(f"IMPORT ERROR:\n{str(e)}")

    st.stop()


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

        # -----------------------------
        # READ PDF
        # -----------------------------

        st.write("READING PDF...")

        text = load_pdf(uploaded_file)

        st.success("PDF LOADED")


        # -----------------------------
        # CHUNKING
        # -----------------------------

        chunks = chunk_text(text)

        st.success(
            f"{len(chunks)} CHUNKS CREATED"
        )


        # -----------------------------
        # EMBEDDINGS
        # -----------------------------

        with st.spinner(
            "CREATING EMBEDDINGS..."
        ):

            embeddings = get_embeddings(
                chunks
            )

        st.success("EMBEDDINGS CREATED")


        # -----------------------------
        # QDRANT COLLECTION
        # -----------------------------

        collection_name = "rag_collection"

        init_qdrant_collection(
            collection_name=collection_name
        )

        st.success(
            "QDRANT COLLECTION CREATED"
        )


        # -----------------------------
        # UPLOAD TO QDRANT
        # -----------------------------

        upload_to_qdrant(
            collection_name,
            chunks,
            embeddings,
        )

        st.success(
            "DATA UPLOADED TO QDRANT"
        )


        # ===================================================
        # CHATBOT
        # ===================================================

        st.divider()

        st.subheader("CHAT WITH PDF")


        question = st.text_input(
            "Ask a question"
        )


        if question:

            with st.spinner("THINKING..."):

                # -----------------------------
                # QUESTION EMBEDDING
                # -----------------------------

                question_embedding = (
                    get_embeddings([question])[0]
                )


                # -----------------------------
                # RETRIEVE CHUNKS
                # -----------------------------

                retrieved_chunks = (
                    retrieve_chunks(
                        collection_name,
                        question_embedding,
                        top_k=4,
                    )
                )


                # -----------------------------
                # ASK LLM
                # -----------------------------

                answer = ask_groq(
                    question,
                    retrieved_chunks,
                )


            # ===================================================
            # SHOW ANSWER
            # ===================================================

            st.subheader("ANSWER")

            st.write(answer)


            # ===================================================
            # SHOW RETRIEVED CHUNKS
            # ===================================================

            with st.expander(
                "VIEW RETRIEVED CHUNKS"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1,
                ):

                    st.write(
                        f"CHUNK {i}"
                    )

                    st.info(chunk)


    except Exception as e:

        st.error(str(e))