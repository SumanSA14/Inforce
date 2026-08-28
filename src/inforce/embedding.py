"""Embeddings via the local Ollama server.

Why Ollama rather than sentence-transformers or fastembed: huggingface.co is
unreachable from this machine (connection reset on plain `requests`, not a
client-library bug), so anything that pulls weights from the HF Hub cannot be
installed. Ollama pulls from its own registry, which works, and
`nomic-embed-text` was already present locally.

Measured throughput on an RTX 3050 4 GB: ~14 chunks/s warm. Concurrency does
not help (12.1 -> 13.9 chunks/s from 1 to 8 workers), so Ollama is already
batching internally and the GPU is the bottleneck. Batch, don't parallelise.
"""
from __future__ import annotations

import os
import time

import numpy as np
import requests

# 127.0.0.1, never "localhost". On Windows, "localhost" resolves to ::1 first;
# Ollama binds IPv4-only, so every request paid a ~2.05s IPv6 connect stall
# before falling back. Measured: 2236ms via localhost vs 176ms via 127.0.0.1
# for an identical single-query embed. That tax applied to all 3,033 batches of
# the M3 index build (~1.7h of pure connection overhead) and put 2.2s of the
# ~3.8s API search latency into nothing but socket setup.
OLLAMA_BASE = os.environ.get("INFORCE_OLLAMA", "http://127.0.0.1:11434")

# One pooled session for the process. Even against 127.0.0.1 a fresh TCP
# handshake per call costs ~40ms; reuse takes the warm embed to ~155ms.
SESSION = requests.Session()
SESSION.mount("http://", requests.adapters.HTTPAdapter(
    pool_connections=4, pool_maxsize=8, max_retries=0))
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768
BATCH_SIZE = 32

# nomic-embed-text is trained with task prefixes; omitting them measurably
# degrades retrieval. A baseline may be simple, but it must not be misused —
# a crippled baseline would inflate N and make the headline result dishonest.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingError(RuntimeError):
    pass


def _post(inputs: list[str], *, model: str, timeout: int, retries: int) -> list[list[float]]:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.post(
                f"{OLLAMA_BASE}/api/embed",
                json={"model": model, "input": inputs},
                timeout=timeout,
            )
            resp.raise_for_status()
            vectors = resp.json().get("embeddings")
            if not vectors or len(vectors) != len(inputs):
                raise EmbeddingError(
                    f"expected {len(inputs)} embeddings, got "
                    f"{0 if not vectors else len(vectors)}"
                )
            return vectors
        except Exception as exc:  # noqa: BLE001 - retried, re-raised below
            last = exc
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise EmbeddingError(f"embedding failed after {retries} attempts: {last}") from last


def embed_documents(
    texts: list[str],
    *,
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
    timeout: int = 600,
    retries: int = 3,
    progress=None,
) -> np.ndarray:
    """Embed chunk texts. Returns an L2-normalised float32 matrix (n, dims)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [DOC_PREFIX + t for t in texts[i : i + batch_size]]
        out.extend(_post(batch, model=model, timeout=timeout, retries=retries))
        if progress:
            progress(min(i + batch_size, len(texts)), len(texts))
    return l2_normalise(np.asarray(out, dtype=np.float32))


def embed_query(
    text: str, *, model: str = EMBED_MODEL, timeout: int = 120, retries: int = 3
) -> np.ndarray:
    vec = _post([QUERY_PREFIX + text], model=model, timeout=timeout, retries=retries)[0]
    return l2_normalise(np.asarray([vec], dtype=np.float32))[0]


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Normalising once at write time turns cosine similarity into a plain dot
    product at query time — one matmul over the whole index."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def health() -> tuple[bool, str]:
    try:
        resp = SESSION.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
        if EMBED_MODEL.split(":")[0] not in names:
            return False, f"{EMBED_MODEL} not pulled — run: ollama pull {EMBED_MODEL}"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"ollama unreachable at {OLLAMA_BASE}: {exc}"
