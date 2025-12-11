# rag-demo-
# Minimal RAG (Retrieval-Augmented Generation) Demo

#This project is a small, educational implementation of a Retrieval-Augmented Generation (RAG) pipeline in Python.

#The goal is to explore how retrieval quality and document embeddings affect downstream question-answering,
#and to provide a clean, easy-to-read codebase that can be extended with real LLMs.

## Features

#- Builds an embedding index over a small sample corpus using `sentence-transformers`
#- Uses `k`-nearest neighbors with cosine distance for similarity search
#- Retrieves the top-k most relevant chunks for a user query
#- Includes a simple generator stub where an LLM API can be plugged in
#- Command-line interface for interactive querying

## Tech Stack

#- Python
##- scikit-learn (NearestNeighbors)
#- NumPy

## Getting Started

#```bash
#git clone https://github.com/<your-username>/rag-demo.git
#cd rag-demo

#python -m venv .venv
#source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

python rag_app.py

rag-demo/
rag-demo/
│
├── README.md
├── requirements.txt
└── rag_app.py
sentence-transformers==3.0.1
scikit-learn==1.5.1
numpy==2.0.0
"""
Minimal RAG (Retrieval-Augmented Generation) demo.

- Build an embedding index over a small document corpus
- Retrieve top-k relevant chunks for a user query
- (Optional) Feed retrieved context into an LLM

Author: Bekzat Gulzhigit uulu
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Data model
# -----------------------------
@dataclass
class DocumentChunk:
    id: int
    source: str
    text: str


# -----------------------------
# Index / Retriever
# -----------------------------
class RAGIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.nn = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, chunks: List[DocumentChunk], n_neighbors: int = 5) -> None:
        """Embed all chunks and build a kNN index."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        print(f"[RAGIndex] Encoding {len(texts)} chunks...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        self.nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="auto",
        )
        self.nn.fit(self.embeddings)
        print("[RAGIndex] Index built.")

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Return top-k most similar chunks for a given query."""
        if self.nn is None or self.embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.nn.kneighbors(query_vec, n_neighbors=k)

        results: List[Tuple[DocumentChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[int(idx)]
            similarity = 1.0 - float(dist)  # cosine distance -> similarity
            results.append((chunk, similarity))

        return results


# -----------------------------
# Generator (LLM stub)
# -----------------------------
def generate_answer(query: str, retrieved: List[Tuple[DocumentChunk, float]]) -> str:
    """
    Very simple 'generator' that just concatenates context.

    In a real project, you would call an LLM API here (OpenAI, etc.),
    passing the query + retrieved context as the prompt.
    """
    context_blocks = []
    for chunk, score in retrieved:
        context_blocks.append(f"[source={chunk.source}, score={score:.3f}]\n{chunk.text}")

    context_text = "\n\n".join(context_blocks)

    # Placeholder answer logic
    answer = (
        "This is a minimal demo answer.\n\n"
        "Query:\n"
        f"{query}\n\n"
        "Retrieved context:\n"
        f"{context_text}\n\n"
        "In a full implementation, an LLM would now generate a synthesized response based "
        "on the query and the context above."
    )
    return answer


# -----------------------------
# Example corpus
# -----------------------------
def build_sample_corpus() -> List[DocumentChunk]:
    """Create a tiny in-memory corpus. Replace with your own documents later."""
    raw_docs = [
        (
            "fusion_overview.txt",
            "Fusion energy is generated when two light atomic nuclei combine to form a heavier nucleus, "
            "releasing a large amount of energy in the process.",
        ),
        (
            "stellarators.txt",
            "Stellarators are a type of fusion device that use twisted magnetic fields to confine hot plasma. "
            "They offer steady-state operation but are geometrically complex.",
        ),
        (
            "tokamaks_vs_stellarators.txt",
            "Tokamaks and stellarators are both magnetic confinement fusion concepts. "
            "Tokamaks are simpler but require pulsed operation and complex control systems, "
            "while stellarators are intrinsically steady-state.",
        ),
        (
            "cs_rag.txt",
            "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models. "
            "The retriever selects relevant documents, and the generator uses them as context.",
        ),
    ]

    chunks: List[DocumentChunk] = []
    for i, (source, text) in enumerate(raw_docs):
        chunks.append(DocumentChunk(id=i, source=source, text=text))
    return chunks


# -----------------------------
# CLI entry point
# -----------------------------
def main() -> None:
    print("=== Minimal RAG Demo ===")
    print("Building index over sample corpus...\n")

    corpus = build_sample_corpus()
    index = RAGIndex()
    index.build_index(corpus)

    while True:
        try:
            query = input("\nEnter your question (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        retrieved = index.retrieve(query, k=3)
        print("\nTop retrieved chunks:")
        for chunk, score in retrieved:
            print(f"- [{score:.3f}] {chunk.source}: {chunk.text[:80]}...")

        answer = generate_answer(query, retrieved)
        print("\n--- Generated Answer (demo) ---")
        print(answer)


if __name__ == "__main__":
    main()
