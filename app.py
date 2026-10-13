
import io
import re
from pathlib import Path

import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 3
MIN_SIMILARITY = 0.05


# -------------------- DOCUMENT PARSING --------------------

def extract_text(uploaded_file):
    """Extract text from supported document formats."""
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    try:
        if filename.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="replace")

        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(
                        f"[Page {page_number}]\n{text}"
                    )

            return "\n\n".join(pages)

        if filename.endswith(".docx"):
            document = Document(io.BytesIO(file_bytes))
            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]
            return "\n".join(paragraphs)

    except Exception as exc:
        raise ValueError(
            f"Could not read {uploaded_file.name}: {exc}"
        ) from exc

    raise ValueError("Unsupported file format.")


# -------------------- TEXT CHUNKING --------------------

def split_into_chunks(text):
    """Split text into overlapping character-based chunks."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# -------------------- RETRIEVAL --------------------

def build_index(documents):
    """Build a TF-IDF index for all document chunks."""
    chunks = []

    for document_name, text in documents.items():
        for chunk_number, chunk_text in enumerate(
            split_into_chunks(text), start=1
        ):
            chunks.append({
                "document": document_name,
                "chunk_number": chunk_number,
                "text": chunk_text,
            })

    if not chunks:
        return [], None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )
    except ValueError:
        return [], None, None

    return chunks, vectorizer, matrix


def retrieve_chunks(question, chunks, vectorizer, matrix):
    """Return the most relevant chunks for a question."""
    question = question.strip()

    if not question or vectorizer is None or matrix is None:
        return []

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector, matrix
    ).flatten()

    ranked_indices = scores.argsort()[::-1]
    results = []

    for index in ranked_indices[:TOP_K]:
        score = float(scores[index])

        if score < MIN_SIMILARITY:
            continue

        result = chunks[index].copy()
        result["score"] = score
        results.append(result)

    return results


# -------------------- GROUNDED ANSWER --------------------

def split_sentences(text):
    """Split a chunk into candidate answer sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def generate_answer(question, retrieved_chunks, vectorizer):
    """
    Produce an extractive answer using retrieved document text.

    This version does not call an external LLM. The answer is
    selected from the source text to avoid invented information.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find sufficiently relevant information "
            "in the uploaded documents. Try rephrasing your question "
            "or uploading a document that covers this topic."
        )

    candidates = []

    for chunk in retrieved_chunks:
        for sentence in split_sentences(chunk["text"]):
            if len(sentence.split()) < 3:
                continue

            candidates.append({
                "sentence": sentence,
                "document": chunk["document"],
                "chunk_number": chunk["chunk_number"],
            })

    if not candidates:
        return (
            "Relevant text was retrieved, but I couldn't identify "
            "a suitable answer sentence. Please inspect the sources below."
        )

    candidate_texts = [item["sentence"] for item in candidates]
    candidate_matrix = vectorizer.transform(candidate_texts)
    question_vector = vectorizer.transform([question])

    sentence_scores = cosine_similarity(
        question_vector, candidate_matrix
    ).flatten()

    best_index = int(sentence_scores.argmax())

    if sentence_scores[best_index] < MIN_SIMILARITY:
        return (
            "I couldn't find a sufficiently relevant answer "
            "in the uploaded documents."
        )

    best = candidates[best_index]

    return (
        f'{best["sentence"]}\n\n'
        f'Source: {best["document"]}, '
        f'chunk {best["chunk_number"]}.'
    )


# -------------------- STREAMLIT INTERFACE --------------------

st.title("📚 Document Q&A Assistant")
st.caption(
    "Upload documents, ask questions, and inspect the source passages."
)

st.info(
    "This prototype uses TF-IDF retrieval and extractive answers. "
    "It does not use an external language model or send documents "
    "to an AI API."
)

uploaded_files = st.file_uploader(
    "Upload TXT, PDF, or DOCX documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Process documents", type="primary"):
        with st.spinner("Reading documents and building the index..."):
            documents = {}
            errors = []

            for uploaded_file in uploaded_files:
                try:
                    text = extract_text(uploaded_file)

                    if text.strip():
                        documents[uploaded_file.name] = text
                    else:
                        errors.append(
                            f"{uploaded_file.name}: no extractable text found."
                        )

                except Exception as exc:
                    errors.append(str(exc))

            chunks, vectorizer, matrix = build_index(documents)

            st.session_state["documents"] = documents
            st.session_state["chunks"] = chunks
            st.session_state["vectorizer"] = vectorizer
            st.session_state["matrix"] = matrix
            st.session_state["processed_files"] = list(documents.keys())

        for error in errors:
            st.warning(error)

        if chunks:
            st.success(
                f"Processed {len(documents)} document(s) "
                f"and created {len(chunks)} searchable chunks."
            )
        else:
            st.error(
                "No searchable text was found. Check your files "
                "and try again."
            )

if st.session_state.get("chunks"):
    st.divider()
    st.subheader("Ask your documents")

    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="What are the main findings in this document?",
        )
        submitted = st.form_submit_button("Find answer")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            results = retrieve_chunks(
                question,
                st.session_state["chunks"],
                st.session_state["vectorizer"],
                st.session_state["matrix"],
            )

            answer = generate_answer(
                question,
                results,
                st.session_state["vectorizer"],
            )

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Retrieved source passages")

            if results:
                for rank, result in enumerate(results, start=1):
                    with st.expander(
                        f"Source {rank}: {result['document']} "
                        f"(chunk {result['chunk_number']})"
                    ):
                        st.caption(
                            f"TF-IDF cosine similarity: "
                            f"{result['score']:.3f}"
                        )
                        st.write(result["text"])
            else:
                st.caption("No source passages passed the relevance threshold.")

    st.sidebar.subheader("Indexed documents")
    for filename in st.session_state.get("processed_files", []):
        st.sidebar.write(f"📄 {filename}")

    if st.sidebar.button("Clear indexed documents"):
        for key in (
            "documents",
            "chunks",
            "vectorizer",
            "matrix",
            "processed_files",
        ):
            st.session_state.pop(key, None)

        st.rerun()
else:
    st.markdown("### Getting started")
    st.write(
        "1. Upload one or more supported documents.\n"
        "2. Click **Process documents**.\n"
        "3. Enter a question about their contents.\n"
        "4. Review the answer and its retrieved source passages."
    )
