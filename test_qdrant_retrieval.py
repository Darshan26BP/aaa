import os
import re
import sys
import json
from typing import List, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


def extract_source_url(text: str) -> str:
    match = re.search(r"\[Source:\s*([^\]]+)\]", text or "")
    return match.group(1).strip() if match else ""


def retrieve_top_k(query: str, k: int = 5) -> List[Tuple[float, str, str]]:
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    collection = os.getenv("QDRANT_COLLECTION", "spdy-chatbot").strip()

    if not qdrant_url or not qdrant_api_key:
        raise SystemExit("❌ Set QDRANT_URL and QDRANT_API_KEY in .env before running this script.")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30.0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    vector = embedder.encode(query).tolist()
    results = client.search(collection_name=collection, query_vector=vector, limit=k)

    out: List[Tuple[float, str, str]] = []
    for hit in results:
        text = (hit.payload or {}).get("text", "")
        url = extract_source_url(text)
        score = float(getattr(hit, "score", 0.0))
        out.append((score, text, url))
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test Qdrant retrieval: fetch top-k relevant chunks.")
    parser.add_argument("--query", required=False, default="admission process", help="Query to search for")
    parser.add_argument("--k", type=int, required=False, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")
    args = parser.parse_args()

    try:
        hits = retrieve_top_k(args.query, args.k)
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        sys.exit(1)

    if args.json:
        print(json.dumps([
            {"score": s, "url": u, "text": t}
            for (s, t, u) in hits
        ], ensure_ascii=False, indent=2))
        return

    print(f"\nTop {len(hits)} results for query: '{args.query}':\n")
    for idx, (score, text, url) in enumerate(hits, start=1):
        snippet = (text or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        print(f"{idx}. score={score:.4f}")
        if url:
            print(f"   source: {url}")
        print(f"   text: {snippet}\n")


if __name__ == "__main__":
    main()


