"""Two-PDF RAG: PDF -> chunks -> embeddings -> persistent FAISS -> top-3 -> optional Ollama answer."""

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 130
OVERLAP_WORDS = 25
TOP_K = 3


def read_and_chunk(pdf_paths):
    """Read both PDFs, keeping PDF, page, and chunk number for citations."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        num_chunks = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if not text:
                print(f"Warning: {pdf_path.name}, page {page_num}: no extractable text")
                continue

            words = text.split()
            for start in range(0, len(words), step):
                chunk_words = words[start:start + CHUNK_WORDS]
                if not chunk_words:
                    break
                chunks.append({
                    "source": pdf_path.name,
                    "page": page_num,
                    "chunk": num_chunks + 1,
                    "text": " ".join(chunk_words),
                })
                num_chunks += 1
                if start + CHUNK_WORDS >= len(words):
                    break
        print(f"Read {pdf_path.name}: {len(reader.pages)} pages, {num_chunks} chunks")

    if not chunks:
        raise ValueError("No text extracted. Scanned/image-only PDFs need OCR first.")
    return chunks


def file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def build_or_load_store(pdf_paths, store_dir, model):
    store_dir.mkdir(parents=True, exist_ok=True)
    index_path = store_dir / "index.faiss"
    metadata_path = store_dir / "metadata.json"
    signatures = [{"file": str(p.resolve()), "sha256": file_digest(p)} for p in pdf_paths]
    configuration = {
        "files": signatures,
        "embedding_model": EMBED_MODEL,
        "chunk_words": CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
    }

    if index_path.exists() and metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved.get("config") == configuration:
            index = faiss.read_index(str(index_path))
            chunks = saved["chunks"]
            if index.ntotal == len(chunks):
                print(f"Loaded {index.ntotal} stored vectors from {index_path}")
                return index, chunks

    chunks = read_and_chunk(pdf_paths)
    print("Creating embeddings...")
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Normalized vectors + inner product = cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(
        json.dumps({"config": configuration, "chunks": chunks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Stored {index.ntotal} vectors in {index_path}")
    return index, chunks


def retrieve(question, model, index, chunks, k=TOP_K):
    vector = model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, ids = index.search(vector, min(k, index.ntotal))
    return [
        {"score": float(score), **chunks[int(idx)]}
        for score, idx in zip(scores[0], ids[0])
        if idx >= 0
    ]


def generate_answer(question, results, ollama_model):
    """Optional generation using a locally running Ollama server."""
    context = "\n\n".join(
        f"[{i}] Source: {r['source']}, page {r['page']}\n{r['text']}"
        for i, r in enumerate(results, start=1)
    )
    prompt = (
        "Answer the question using ONLY the provided PDF excerpts. "
        "Cite relevant excerpt numbers in square brackets, e.g. [1]. "
        "If the excerpts do not contain the answer, say: "
        "'I don't know based on these PDFs.'\n\n"
        f"PDF excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": ollama_model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode())["response"]
    except (urllib.error.URLError, TimeoutError, KeyError) as exc:
        return (
            f"Answer generation unavailable ({exc}). "
            f"The top {len(results)} chunks above were retrieved successfully. "
            f"Start Ollama and run `ollama pull {ollama_model}`."
        )


def ask(question, model, index, chunks, generate, ollama_model):
    results = retrieve(question, model, index, chunks)
    print(f"\nQuestion: {question}\nTop {len(results)} retrieved chunks:")
    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] PDF: {result['source']} | Page: {result['page']} "
              f"| Chunk: {result['chunk']} | Cosine similarity: {result['score']:.4f}")
        print(result["text"])
    if generate:
        print("\nGenerated answer:\n" + generate_answer(question, results, ollama_model))


def main():
    parser = argparse.ArgumentParser(description="Search exactly 2 PDFs with FAISS; optionally generate answers")
    parser.add_argument("pdf1", type=Path, help="Path to first PDF")
    parser.add_argument("pdf2", type=Path, help="Path to second PDF")
    parser.add_argument("--question", help="Ask one question; otherwise interactive mode")
    parser.add_argument("--generate", action="store_true", help="Generate an answer with local Ollama")
    parser.add_argument("--ollama-model", default="llama3.2", help="Local Ollama model name")
    parser.add_argument("--store", type=Path, default=Path("vector_store"), help="FAISS database folder")
    args = parser.parse_args()

    pdf_paths = [args.pdf1, args.pdf2]
    if pdf_paths[0].resolve() == pdf_paths[1].resolve():
        parser.error("Please provide two different PDFs.")
    for pdf_path in pdf_paths:
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            parser.error(f"Not an existing PDF: {pdf_path}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    index, chunks = build_or_load_store(pdf_paths, args.store, model)
    if args.question:
        ask(args.question, model, index, chunks, args.generate, args.ollama_model)
    else:
        print("\nAsk questions about your PDFs. Type 'exit' to quit.")
        while True:
            try:
                question = input("\nYour question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                ask(question, model, index, chunks, args.generate, args.ollama_model)


if __name__ == "__main__":
    main()
