# upload_to_qdrant.py
import os
import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv

# ---------- CONFIG (from .env or fallback) ----------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "spdy-chatbot")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "spdy_website_chunks.json")
DIMENSION = 384   # for all-MiniLM-L6-v2
BATCH_SIZE = 50
RETRIES = 3

if not QDRANT_URL or not QDRANT_API_KEY:
    raise SystemExit("❌ Set QDRANT_URL and QDRANT_API_KEY in .env before running this script.")

print("🔄 Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)

# Ensure collection exists
try:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"📦 Creating new collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=DIMENSION, distance=rest.Distance.COSINE),
        )
    else:
        print(f"📦 Using existing collection: {COLLECTION_NAME}")
except Exception as e:
    print("⚠️ Could not ensure collection exists:", e)
    raise

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

if not isinstance(chunks, list):
    raise SystemExit("❌ Chunks file must be a JSON array of strings.")

print(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Build point structs
points = []
for i, chunk in enumerate(tqdm(chunks, desc="Encoding")):
    vec = embedder.encode(chunk).tolist()
    points.append(rest.PointStruct(id=i, vector=vec, payload={"text": chunk}))

# Upload in batches
print(f"⬆️ Uploading {len(points)} points to Qdrant in batches of {BATCH_SIZE}...")
for start in range(0, len(points), BATCH_SIZE):
    batch = points[start : start + BATCH_SIZE]
    for attempt in range(1, RETRIES + 1):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            break
        except Exception as e:
            print(f"⚠️ Upsert attempt {attempt}/{RETRIES} failed: {e}")
            if attempt == RETRIES:
                raise
            time.sleep(2 ** attempt)

print("✅ Done uploading all chunks. Qdrant is now updated with the latest data.")
