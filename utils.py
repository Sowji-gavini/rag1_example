import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------
# PDF
# ---------------------------------------------------

def load_pdf(uploaded_file):

    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ---------------------------------------------------
# CHUNKING
# ---------------------------------------------------

def chunk_text(
    text,
    chunk_size=500,
    overlap=50,
):

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------

def get_embeddings(texts):

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = model.encode(
        texts,
        show_progress_bar=False,
    )

    return vectors.tolist()


# ---------------------------------------------------
# QDRANT INIT
# ---------------------------------------------------

def init_qdrant_collection(
    collection_name,
    vector_size=384,
):

    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
    )

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    existing = [
        c.name
        for c in qdrant.get_collections().collections
    ]

    if collection_name in existing:
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


# ---------------------------------------------------
# QDRANT UPLOAD
# ---------------------------------------------------

def upload_to_qdrant(
    collection_name,
    chunks,
    embeddings,
):

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    points = []

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk
                },
            )
        )

    qdrant.upsert(
        collection_name=collection_name,
        points=points,
    )


# ---------------------------------------------------
# RETRIEVE
# ---------------------------------------------------

def retrieve_chunks(
    collection_name,
    question_embedding,
    top_k=4,
):

    from qdrant_client import QdrantClient

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    results = qdrant.query_points(
        collection_name=collection_name,
        query=question_embedding,
        limit=top_k,
    ).points

    chunks = []

    for hit in results:

        chunks.append(
            hit.payload["text"]
        )

    return chunks


# ---------------------------------------------------
# GROQ
# ---------------------------------------------------

def ask_groq(
    question,
    context_chunks,
):

    from groq import Groq

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )

    context = "\n\n".join(context_chunks)

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content":
                "Answer ONLY using the provided context."
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
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content
