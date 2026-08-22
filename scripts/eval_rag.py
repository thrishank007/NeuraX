"""
RAG retrieval evaluation: quality metrics + latency benchmark.

Modes: dense (NIM), bm25, hybrid (RRF fusion).
Metrics: precision@k, recall@k, MRR against hand-labeled golden set.
Usage: venv\\Scripts\\python.exe scripts\\eval_rag.py [--k 5] [--runs 5]
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import numpy as np

from config import NIM_EMBEDDING_CONFIG
from indexing.nvidia_nim_embedding_provider import NvidiaNimEmbeddingProvider

COLLECTION_NAME = NIM_EMBEDDING_CONFIG["collection_name"]

# Golden set: query -> set of relevant chunk-id prefixes (12-char).
# Labeled by hand against live corpus. Relabeled 2026-08-22 after the
# chunker rotation fix (indexing/text_chunker.py): CV went from 104
# rotation slivers to 6 real chunks, offer letter gained idx 0-1.
GOLDEN_SET = [
    {
        "query": "What is the GSTIN number of the company?",
        "relevant": ["9e285bf5223d", "24c07d940faf", "cdc86ac7ddd0"],
    },
    {
        "query": "Where is the candidate located?",
        "relevant": ["71b730c8159f", "516ddfe69b15"],
    },
    {
        "query": "Who is the managing director?",
        "relevant": ["71b730c8159f"],
    },
    {
        "query": "What MSME registration does the company have?",
        "relevant": ["9e285bf5223d", "24c07d940faf", "cdc86ac7ddd0"],
    },
    {
        "query": "What contact email and phone number are listed?",
        "relevant": ["9e285bf5223d", "24c07d940faf", "cdc86ac7ddd0", "516ddfe69b15"],
    },
    {
        "query": "What agentic AI skills does the candidate have?",
        "relevant": ["63bd8c8c1311"],
    },
    {
        "query": "Tell me about the geolocation routing engine",
        "relevant": ["38caffae3e8b"],
    },
    {
        "query": "What backend stack was used for the platform project?",
        "relevant": ["38caffae3e8b"],
    },
    {
        "query": "What data security clause exists in the offer letter?",
        "relevant": ["26c7bdd8748c"],
    },
    {
        "query": "What multimodal pipeline did the candidate engineer?",
        "relevant": ["d93a4d3ba708", "cbd62d23b439"],
    },
    {
        "query": "What is the effective date mentioned in the letter?",
        "relevant": ["71b730c8159f"],
    },
    {
        "query": "How is token usage tracked across models?",
        "relevant": ["cbd62d23b439"],
    },
]

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def load_collection(name: str | None = None, golden: dict | None = None):
    collection = name or (golden or {}).get("collection") or COLLECTION_NAME
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "vector_db"))
    col = client.get_collection(collection)
    return col


class Retriever:
    def __init__(self, col, embedder):
        self.col = col
        self.embedder = embedder
        self._dense_cache: dict[str, list[float]] = {}
        data = col.get(include=["documents", "metadatas"])
        self.ids = list(data["ids"])
        self.docs = [d or "" for d in data["documents"]]
        self.metas = data["metadatas"]
        self._doc_by_id = dict(zip(self.ids, self.docs))
        self._bm25 = None
        self._reranker = None
        from config import NIM_RERANK_CONFIG
        if NIM_RERANK_CONFIG.get("enabled") and NIM_RERANK_CONFIG.get("api_key"):
            try:
                from retrieval.nim_reranker import NimReranker
                self._reranker = NimReranker(NIM_RERANK_CONFIG)
            except Exception as e:
                print(f"reranker unavailable, 'rerank' mode falls back to hybrid_w: {e}")

    def _embed(self, query: str) -> list[float]:
        if query not in self._dense_cache:
            last_err = None
            for attempt in range(3):
                try:
                    vec = np.asarray(self.embedder.embed_query(query)).reshape(-1)
                    self._dense_cache[query] = vec.tolist()
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"NIM embed failed after 3 retries: {last_err}")
        return self._dense_cache[query]

    def _build_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(d) for d in self.docs])
        return self._bm25

    def dense_search(self, query: str, k: int) -> list[str]:
        res = self.col.query(query_embeddings=[self._embed(query)], n_results=min(k * 3, len(self.ids)))
        return res["ids"][0][:k]

    def bm25_search(self, query: str, k: int) -> list[str]:
        bm25 = self._build_bm25()
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:k] if scores[i] > 0]

    def hybrid_search(self, query: str, k: int, rrf_k: int = 60,
                      w_dense: float = 1.0, w_sparse: float = 1.0) -> list[str]:
        dense = self.dense_search(query, k=20)
        sparse = self.bm25_search(query, k=20)
        fused: dict[str, float] = {}
        for rank, cid in enumerate(dense):
            fused[cid] = fused.get(cid, 0.0) + w_dense / (rrf_k + rank + 1)
        for rank, cid in enumerate(sparse):
            fused[cid] = fused.get(cid, 0.0) + w_sparse / (rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:k]]

    def rerank_search(self, query: str, k: int) -> list[str]:
        """Production pipeline: weighted-RRF fusion -> NIM cross-encoder rerank."""
        candidates = self.hybrid_search(query, k=20, rrf_k=20, w_dense=0.9, w_sparse=0.1)
        if self._reranker is None or len(candidates) <= 1:
            return candidates[:k]
        texts = [self._doc_by_id[cid] for cid in candidates]
        try:
            order = self._reranker.rerank(query, texts)
        except Exception as e:
            print(f"rerank API failed, keeping fusion order: {e}")
            return candidates[:k]
        return [candidates[i] for i in order[:k]]


def match_ids(retrieved: list[str], prefixes: list[str]) -> list[str]:
    out = []
    for cid in retrieved:
        for p in prefixes:
            if cid.startswith(p):
                out.append(p)
                break
    return out


def evaluate_mode(retriever: Retriever, mode: str, k: int, golden_set: list[dict]):
    precisions, recalls, rranks, per_query = [], [], [], []
    for item in golden_set:
        q = item["query"]
        rel_prefixes = item["relevant"]
        if mode == "dense":
            retrieved = retriever.dense_search(q, k)
        elif mode == "bm25":
            retrieved = retriever.bm25_search(q, k)
        elif mode == "hybrid_w":
            # production-weighted RRF (dense 0.9 / bm25 0.1, rrf_k 20) —
            # mirrors retrieval/query_processor.py defaults
            retrieved = retriever.hybrid_search(q, k, rrf_k=20, w_dense=0.9, w_sparse=0.1)
        elif mode == "rerank":
            retrieved = retriever.rerank_search(q, k)
        else:
            retrieved = retriever.hybrid_search(q, k)

        hits = match_ids(retrieved, rel_prefixes)
        n_unique_hits = len(set(hits))
        n_rel = len(set(rel_prefixes))
        precisions.append(n_unique_hits / k)
        recalls.append(n_unique_hits / n_rel)
        rr = 0.0
        seen = set()
        for pos, h in enumerate(hits, start=1):
            if h not in seen:
                rr = 1.0 / pos
                break
        rranks.append(rr)
        per_query.append({
            "query": q,
            "class": item.get("class"),
            "hits": len(hits),
            "unique_hits": n_unique_hits,
            "relevant": n_rel,
            "reciprocal_rank": round(rr, 4),
        })
    by_class = {}
    for entry in per_query:
        cls = entry.get("class") or "unlabeled"
        bucket = by_class.setdefault(cls, {"p": [], "r": [], "hit": []})
        bucket["p"].append(min(entry["unique_hits"], entry["relevant"]) / k)
        bucket["r"].append(min(entry["unique_hits"], entry["relevant"]) / entry["relevant"])
        bucket["hit"].append(1.0 if entry["reciprocal_rank"] > 0 else 0.0)
    class_summary = {
        cls: {
            "n": len(b["hit"]),
            "recall": round(statistics.mean(b["r"]), 3),
            "hit_rate": round(statistics.mean(b["hit"]), 3),
        }
        for cls, b in by_class.items()
    }
    return {
        "mode": mode,
        f"precision@{k}": statistics.mean(precisions),
        f"recall@{k}": statistics.mean(recalls),
        "mrr": statistics.mean(rranks),
        "queries_with_hit": sum(1 for r in rranks if r > 0),
        "total_queries": len(golden_set),
        "per_class": class_summary,
        "per_query": per_query,
    }


def bench_latency(retriever: Retriever, mode: str, k: int, runs: int) -> dict:
    """Search-only latency (embeddings pre-cached). Warmup 1 run."""
    q = GOLDEN_SET[0]["query"]
    fn = {
        "dense": lambda: retriever.dense_search(q, k),
        "bm25": lambda: retriever.bm25_search(q, k),
        "hybrid": lambda: retriever.hybrid_search(q, k),
        "hybrid_w": lambda: retriever.hybrid_search(q, k, rrf_k=20, w_dense=0.9, w_sparse=0.1),
        "rerank": lambda: retriever.rerank_search(q, k),
    }[mode]
    fn()  # warmup
    samples_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000)
    ordered = sorted(samples_ms)
    return {
        "mode": mode,
        "p50_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 1),
        "mean_ms": round(statistics.mean(ordered), 1),
    }


def bench_embed_latency(embedder, runs: int) -> dict:
    q = GOLDEN_SET[0]["query"]
    embedder.embed_query(q)  # warmup
    samples_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        embedder.embed_query(q)
        samples_ms.append((time.perf_counter() - t0) * 1000)
    ordered = sorted(samples_ms)
    return {"p50_ms": round(statistics.median(ordered), 1), "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 1)}


def corpus_health(retriever: Retriever) -> dict:
    """Detect degenerate near-duplicate / tiny chunks."""
    titles: dict[str, int] = {}
    tiny = 0
    texts = [d.strip() for d in retriever.docs]
    for d, m in zip(texts, retriever.metas):
        t = m.get("title", "<none>")
        titles[t] = titles.get(t, 0) + 1
        if len(d) < 120:
            tiny += 1
    # near-dup: chunk text is substring of another chunk (rotation artifacts)
    dup_like = 0
    sorted_by_len = sorted(texts, key=len, reverse=True)
    for i, d in enumerate(sorted_by_len):
        if len(d) < 60:
            continue
        if any(d in other for other in sorted_by_len[:i] ):
            dup_like += 1
    return {
        "collection": retriever.col.name,
        "total_chunks": len(texts),
        "chunks_per_title": titles,
        "tiny_chunks_lt120chars": tiny,
        "substring_duplicate_chunks": dup_like,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--golden", type=str, default=None,
                        help="Path to golden JSON (e.g. scripts/eval_golden_synthetic.json). "
                             "Default: built-in GOLDEN_SET against the live collection.")
    parser.add_argument("--collection", type=str, default=None,
                        help="Override collection name (default: golden's collection or live).")
    args = parser.parse_args()

    golden_doc = None
    golden_set = GOLDEN_SET
    if args.golden:
        golden_doc = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        golden_set = golden_doc["queries"]

    col = load_collection(args.collection, golden_doc)
    embedder = NvidiaNimEmbeddingProvider(NIM_EMBEDDING_CONFIG)
    retriever = Retriever(col, embedder)
    if not retriever.ids:
        print(f"FATAL: collection '{col.name}' empty. Ingest docs first.")
        sys.exit(1)

    print(f"=== RAG Evaluation ===")
    print(f"Collection: {col.name} ({len(retriever.ids)} chunks)")
    print(f"Golden: {args.golden or 'built-in GOLDEN_SET'} ({len(golden_set)} queries)")

    health = corpus_health(retriever)
    print("\n--- Corpus Health ---")
    print(json.dumps(health, indent=2))

    print(f"\n--- Quality (k={args.k}) ---")
    quality = []
    for mode in ["dense", "bm25", "hybrid", "hybrid_w", "rerank"]:
        r = evaluate_mode(retriever, mode, args.k, golden_set)
        quality.append(r)
        print(f"{mode:8s} P@{args.k}={r[f'precision@{args.k}']:.3f} "
              f"R@{args.k}={r[f'recall@{args.k}']:.3f} "
              f"MRR={r['mrr']:.3f} hit-rate={r['queries_with_hit']}/{r['total_queries']}")
        for cls, s in sorted(r["per_class"].items()):
            if cls != "unlabeled":
                print(f"         {cls:10s} R@{args.k}={s['recall']:.3f} hit={s['hit_rate']:.2f} (n={s['n']})")

    print(f"\n--- Latency (search only, {args.runs} runs) ---")
    latency = []
    for mode in ["dense", "bm25", "hybrid", "hybrid_w", "rerank"]:
        l = bench_latency(retriever, mode, args.k, args.runs)
        latency.append(l)
        print(f"{l['mode']:8s} p50={l['p50_ms']}ms p95={l['p95_ms']}ms mean={l['mean_ms']}ms")

    emb_lat = bench_embed_latency(embedder, args.runs)
    print(f"\nembed_query (NIM API): p50={emb_lat['p50_ms']}ms p95={emb_lat['p95_ms']}ms")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": args.k,
        "collection": col.name,
        "golden": args.golden or "builtin",
        "corpus_health": health,
        "quality": [{key: val for key, val in q.items() if key != "per_query"} for q in quality],
        "quality_detail": quality,
        "latency": latency,
        "embedding_latency": emb_lat,
    }
    out_path = PROJECT_ROOT / "logs" / f"rag_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
