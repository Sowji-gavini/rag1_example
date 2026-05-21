import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ===================================================
# LOAD PDF
# ===================================================

def load_pdf(uploaded_file):

    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ===================================================
# CHUNK TEXT
# ===================================================

def chunk_text(
    text,
    chunk_size=500,
    overlap=50,
):

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ===================================================
# EMBEDDINGS
# ===================================================

def get_embedding_model():

    from sentence_transformers import (
        SentenceTransformer
    )

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


# ===================================================
# CREATE VECTORSTORE
# ===================================================

def create_vectorstore(
    chunks,
    model,
):

    import faiss
    import numpy as np

    embeddings = model.encode(chunks)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


# ===================================================
# RETRIEVE CHUNKS
# ===================================================

def retrieve_chunks(
    question,
    chunks,
    model,
    index,
    top_k=4,
):

    import numpy as np

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k,
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(
            chunks[idx]
        )

    return retrieved_chunks


# ===================================================
# ASK GROQ
# ===================================================

def ask_groq(
    question,
    context_chunks,
):

    from groq import Groq

    client = Groq(
        api_key=GROQ_API_KEY
    )

    context = "\n\n".join(
        context_chunks
    )

    response = client.chat.completions.create(

        model="llama3-8b-8192",

        messages=[

            {
                "role": "system",
                "content":
                "Answer ONLY using provided context."
            },

            {
                "role": "user",
                "content":
                f"""
Context:
{context}

Question:
{question}

Answer:
"""
            }
        ],

        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content