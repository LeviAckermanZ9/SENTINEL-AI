"""
SENTINEL-AI — Knowledge Base Builder
Covers: CSR322 CO5 (RAG system design)

Builds ChromaDB vector store from LIAR, FEVER, and optional Snopes datasets.
Sliding window chunking with SentenceTransformer embeddings.
"""

import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    min_chunk_words: int = 30,
) -> List[str]:
    """
    Split text into overlapping chunks using sliding window.

    Args:
        text: Input text to chunk.
        chunk_size: Target chunk size in words.
        overlap: Word overlap between consecutive chunks.
        min_chunk_words: Skip chunks shorter than this.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if len(words) < min_chunk_words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        if len(chunk_words) >= min_chunk_words:
            chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break
        start += step

    return chunks


def _format_liar_entry(statement: str, label: str, justification: str = "") -> str:
    """Format a LIAR dataset entry into structured text."""
    verdict_map = {
        "false": "FALSE",
        "pants-fire": "FALSE",
        "barely-true": "MOSTLY FALSE",
        "half-true": "MIXED",
        "mostly-true": "MOSTLY TRUE",
        "true": "TRUE",
    }
    verdict = verdict_map.get(str(label).lower(), str(label).upper())
    evidence = (
        justification.strip()
        if justification
        else "No additional justification provided."
    )
    return f"CLAIM: {statement.strip()} VERDICT: {verdict} EVIDENCE: {evidence}"


def _download_liar_tsv(split: str, raw_dir: str) -> str:
    """
    Download a LIAR dataset split TSV directly from HuggingFace.
    Returns the local file path.
    """
    filename = f"liar_{split}.tsv"
    local_path = os.path.join(raw_dir, filename)

    if os.path.exists(local_path):
        print(f"    Using cached {filename}")
        return local_path

    # Direct TSV download URLs for LIAR dataset
    urls = [
        f"https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/data/{split}.tsv",
        f"https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/{split}.tsv",
    ]

    for url in urls:
        try:
            print(f"    Downloading {filename} from {url[:60]}...")
            urllib.request.urlretrieve(url, local_path)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
                print(
                    f"    Downloaded {filename} ({os.path.getsize(local_path)//1024}KB)"
                )
                return local_path
        except Exception as e:
            print(f"    URL failed: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            continue

    return None


def ingest_liar_plus(
    collection,
    embedding_fn,
    batch_size: int = 100,
    chunk_size: int = 300,
    overlap: int = 50,
) -> int:
    """
    Ingest LIAR dataset into ChromaDB by downloading TSV files directly.
    Bypasses the HuggingFace datasets loader which no longer supports
    script-based datasets.

    Combines statement + justification into structured format:
    'CLAIM: ... VERDICT: ... EVIDENCE: ...'

    Args:
        collection: ChromaDB collection.
        embedding_fn: Function to embed text -> vector.
        batch_size: Batch size for ChromaDB upserts.
        chunk_size: Chunk size in words.
        overlap: Chunk overlap in words.

    Returns:
        Number of chunks added.
    """
    import pandas as pd

    raw_dir = "./data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # LIAR TSV column names (no header row in the files)
    cols = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state",
        "party",
        "barely_true_count",
        "false_count",
        "half_true_count",
        "mostly_true_count",
        "pants_on_fire_count",
        "context",
        "justification",
    ]

    total_chunks = 0
    doc_ids, doc_texts, doc_metas = [], [], []

    for split in ["train", "valid", "test"]:
        print(f"  Processing LIAR {split} split...")
        tsv_path = _download_liar_tsv(split, raw_dir)

        if tsv_path is None:
            print(f"  Could not download {split} split — skipping")
            continue

        try:
            df = pd.read_csv(
                tsv_path,
                sep="\t",
                header=None,
                names=cols,
                on_bad_lines="skip",
                dtype=str,
            )
        except Exception as e:
            print(f"  Failed to read {split} TSV: {e}")
            continue

        print(f"  Loaded {len(df)} rows from {split}")

        for _, row in df.iterrows():
            statement = str(row.get("statement", "") or "").strip()
            label = str(row.get("label", "unknown") or "unknown").strip()
            justification = str(row.get("justification", "") or "").strip()
            speaker = str(row.get("speaker", "unknown") or "unknown").strip()

            if not statement or len(statement.split()) < 5:
                continue

            text = _format_liar_entry(statement, label, justification)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

            # If text too short to chunk, add it directly
            if not chunks and len(text.split()) >= 10:
                chunks = [text]

            for j, chunk in enumerate(chunks):
                doc_id = f"liar_{split}_{abs(hash(statement + str(j))) % 10**9}"
                doc_ids.append(doc_id)
                doc_texts.append(chunk)
                doc_metas.append(
                    {
                        "source": "liar",
                        "split": split,
                        "verdict": label,
                        "speaker": speaker[:200],
                        "original_claim": statement[:500],
                    }
                )

                if len(doc_ids) >= batch_size:
                    embeddings = embedding_fn(doc_texts)
                    collection.add(
                        ids=doc_ids,
                        documents=doc_texts,
                        embeddings=embeddings,
                        metadatas=doc_metas,
                    )
                    total_chunks += len(doc_ids)
                    doc_ids, doc_texts, doc_metas = [], [], []

    # Flush remaining
    if doc_ids:
        embeddings = embedding_fn(doc_texts)
        collection.add(
            ids=doc_ids,
            documents=doc_texts,
            embeddings=embeddings,
            metadatas=doc_metas,
        )
        total_chunks += len(doc_ids)

    print(f"  LIAR: added {total_chunks} chunks")
    return total_chunks


def ingest_fever(
    collection,
    embedding_fn,
    sample_size: int = 5000,
    batch_size: int = 100,
    chunk_size: int = 300,
    overlap: int = 50,
) -> int:
    import gzip
    import json

    raw_dir = "./data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    local_path = os.path.join(raw_dir, "fever_train.jsonl")

    # Direct download from FEVER shared task website
    urls = [
        "https://fever.ai/data/fever/train.jsonl",
        "https://raw.githubusercontent.com/awslabs/fever/master/data/fever/train.jsonl",
    ]

    if not os.path.exists(local_path):
        downloaded = False
        for url in urls:
            try:
                print(f"  Downloading FEVER from {url[:60]}...")
                urllib.request.urlretrieve(url, local_path)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 10000:
                    print(f"  Downloaded ({os.path.getsize(local_path)//1024}KB)")
                    downloaded = True
                    break
            except Exception as e:
                print(f"  Failed: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                continue

        if not downloaded:
            print("  FEVER: download failed — skipping")
            return 0

    print(f"  Parsing FEVER JSONL (first {sample_size} rows)...")
    total_chunks = 0
    doc_ids, doc_texts, doc_metas = [], [], []
    count = 0

    label_map = {
        "SUPPORTS": "SUPPORTS",
        "REFUTES": "REFUTES",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
    }

    try:
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= sample_size:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                claim = str(row.get("claim", "") or "").strip()
                label_raw = str(
                    row.get("label", "NOT ENOUGH INFO") or "NOT ENOUGH INFO"
                )
                label_str = label_map.get(label_raw, "NOT ENOUGH INFO")

                if not claim or len(claim.split()) < 5:
                    continue

                text = f"CLAIM: {claim} VERDICT: {label_str}"
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                if not chunks and len(claim.split()) >= 10:
                    chunks = [text]

                for j, chunk in enumerate(chunks):
                    doc_id = f"fever_{count}_{j}"
                    doc_ids.append(doc_id)
                    doc_texts.append(chunk)
                    doc_metas.append(
                        {
                            "source": "fever",
                            "verdict": label_str,
                            "speaker": "unknown",
                            "original_claim": claim[:500],
                        }
                    )

                    if len(doc_ids) >= batch_size:
                        embeddings = embedding_fn(doc_texts)
                        collection.add(
                            ids=doc_ids,
                            documents=doc_texts,
                            embeddings=embeddings,
                            metadatas=doc_metas,
                        )
                        total_chunks += len(doc_ids)
                        doc_ids, doc_texts, doc_metas = [], [], []

                count += 1

    except Exception as e:
        print(f"  FEVER parse error: {e}")

    if doc_ids:
        embeddings = embedding_fn(doc_texts)
        collection.add(
            ids=doc_ids,
            documents=doc_texts,
            embeddings=embeddings,
            metadatas=doc_metas,
        )
        total_chunks += len(doc_ids)

    print(f"  FEVER: added {total_chunks} chunks")
    return total_chunks


def ingest_snopes_csv(
    collection,
    embedding_fn,
    csv_path: str,
    batch_size: int = 100,
    chunk_size: int = 300,
    overlap: int = 50,
) -> int:
    """
    Ingest optional Snopes CSV dataset into ChromaDB.

    Args:
        collection: ChromaDB collection.
        embedding_fn: Embedding function.
        csv_path: Path to Snopes CSV file.
        batch_size: Upsert batch size.

    Returns:
        Number of chunks added.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"  Snopes CSV not found at {csv_path}, skipping.")
        return 0

    print(f"  Loading Snopes CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    total_chunks = 0
    doc_ids, doc_texts, doc_metas = [], [], []

    claim_col = next((c for c in df.columns if "claim" in c.lower()), None)
    verdict_col = next(
        (c for c in df.columns if "verdict" in c.lower() or "rating" in c.lower()), None
    )
    article_col = next(
        (c for c in df.columns if "article" in c.lower() or "content" in c.lower()),
        None,
    )

    if claim_col is None:
        print("  Snopes: no 'claim' column found, skipping.")
        return 0

    for i, row in df.iterrows():
        claim = str(row.get(claim_col, "") or "")
        verdict = str(row.get(verdict_col, "unknown")) if verdict_col else "unknown"
        article = str(row.get(article_col, "")) if article_col else ""

        text = f"CLAIM: {claim} VERDICT: {verdict}"
        if article:
            text += f" EVIDENCE: {article}"

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks and len(claim.split()) >= 10:
            chunks = [text]

        for j, chunk in enumerate(chunks):
            doc_id = f"snopes_{i}_{j}"
            doc_ids.append(doc_id)
            doc_texts.append(chunk)
            doc_metas.append(
                {
                    "source": "snopes",
                    "verdict": verdict,
                    "speaker": "unknown",
                    "original_claim": claim[:500],
                }
            )

            if len(doc_ids) >= batch_size:
                embeddings = embedding_fn(doc_texts)
                collection.add(
                    ids=doc_ids,
                    documents=doc_texts,
                    embeddings=embeddings,
                    metadatas=doc_metas,
                )
                total_chunks += len(doc_ids)
                doc_ids, doc_texts, doc_metas = [], [], []

    if doc_ids:
        embeddings = embedding_fn(doc_texts)
        collection.add(
            ids=doc_ids,
            documents=doc_texts,
            embeddings=embeddings,
            metadatas=doc_metas,
        )
        total_chunks += len(doc_ids)

    print(f"  Snopes: added {total_chunks} chunks")
    return total_chunks


def build_knowledge_base(
    chroma_host: Optional[str] = None,
    chroma_port: Optional[int] = None,
    persist_dir: str = "./data/chromadb",
    embedding_model: str = "all-MiniLM-L6-v2",
    snopes_csv_path: Optional[str] = None,
    eval_mode: bool = False,
    eval_sample_size: int = 1000,
) -> Dict[str, Any]:
    """
    Main entry point: build the SENTINEL-AI fact-check knowledge base.

    Creates/opens ChromaDB collection 'sentinel_factcheck_kb' with cosine
    similarity, then ingests LIAR, FEVER, and optional Snopes data.

    Args:
        chroma_host: ChromaDB server host (None = persistent local client).
        chroma_port: ChromaDB server port.
        persist_dir: Local persistence directory.
        embedding_model: SentenceTransformer model name.
        snopes_csv_path: Optional path to Snopes CSV.
        eval_mode: If True, only ingest eval_sample_size chunks total.
        eval_sample_size: Max chunks in eval mode.

    Returns:
        Dict with total_chunks, collection_name, timings.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("SENTINEL-AI Knowledge Base Builder")
    print("=" * 60)

    # Load embedding model
    print(f"\nLoading embedding model: {embedding_model}")
    embedder = SentenceTransformer(embedding_model)

    def embed_fn(texts: List[str]) -> List[List[float]]:
        vectors = embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    # Connect to ChromaDB
    env_host = chroma_host or os.environ.get("CHROMA_HOST")
    env_port = chroma_port or int(os.environ.get("CHROMA_PORT", "8000"))

    if env_host:
        print(f"Connecting to ChromaDB server at {env_host}:{env_port}")
        chroma_token = os.environ.get("CHROMA_TOKEN", "")
        if chroma_token:
            client = chromadb.HttpClient(
                host=env_host,
                port=env_port,
                headers={"Authorization": f"Bearer {chroma_token}"},
            )
        else:
            client = chromadb.HttpClient(host=env_host, port=env_port)
    else:
        print(f"Using persistent local ChromaDB at {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)

    # Create or get collection
    collection_name = "sentinel_factcheck_kb"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.count()
    print(f"Collection '{collection_name}' ready (existing docs: {existing})")

    if existing > 0:
        print(f"  Collection already has {existing} documents.")
        print("  Delete ./data/chromadb/ to rebuild from scratch.")
        return {
            "total_chunks": existing,
            "collection_name": collection_name,
            "build_time_seconds": 0,
            "liar_chunks": 0,
        }

    start_time = time.time()
    total = 0
    liar_count = 0

    # Ingest LIAR dataset
    print("\n--- Ingesting LIAR dataset ---")
    liar_count = ingest_liar_plus(collection, embed_fn)
    total += liar_count

    # Ingest FEVER dataset
    if not eval_mode or total < eval_sample_size:
        print("\n--- Ingesting FEVER dataset ---")
        fever_sample = eval_sample_size - total if eval_mode else 5000
        fever_count = ingest_fever(
            collection,
            embed_fn,
            sample_size=max(fever_sample, 100),
        )
        total += fever_count

    # Ingest Snopes (optional)
    if snopes_csv_path and (not eval_mode or total < eval_sample_size):
        print("\n--- Ingesting Snopes dataset ---")
        snopes_count = ingest_snopes_csv(collection, embed_fn, snopes_csv_path)
        total += snopes_count

    elapsed = time.time() - start_time
    final_count = collection.count()

    print(f"\n{'=' * 60}")
    print(f"Knowledge Base Build Complete!")
    print(f"  Total chunks indexed: {final_count}")
    print(f"  Time elapsed:         {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return {
        "total_chunks": final_count,
        "collection_name": collection_name,
        "build_time_seconds": elapsed,
        "liar_chunks": liar_count,
    }


# Allow running as: python -m src.rag.knowledge_base
if __name__ == "__main__":
    eval_mode = os.environ.get("RAG_EVAL_MODE", "false").lower() == "true"
    eval_size = int(os.environ.get("RAG_EVAL_SAMPLE_SIZE", "1000"))
    snopes_path = os.environ.get("SNOPES_CSV_PATH")

    build_knowledge_base(
        eval_mode=eval_mode,
        eval_sample_size=eval_size,
        snopes_csv_path=snopes_path,
    )
