# app.py
import os
import re
import time
from flask import Flask, request, jsonify, render_template
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import nltk
from nltk.corpus import wordnet

# ----------------- CONFIG -----------------
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path)
    print(f"✅ Loaded .env from: {_dotenv_path}")
else:
    load_dotenv()
    print("⚠️ No explicit .env file found with find_dotenv; relying on environment variables.")

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "spdy-chatbot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_ORG = os.getenv("OPENAI_ORG", "").strip()
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()

# ----------------- Validate Environment Variables -----------------
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is not set in .env file")
if not QDRANT_URL:
    raise ValueError("❌ QDRANT_URL is not set in .env file")
if not QDRANT_API_KEY:
    raise ValueError("❌ QDRANT_API_KEY is not set in .env file")

# ----------------- Initialize -----------------
app = Flask(__name__)

try:
    # Allow optional organization/project hints (useful for some account setups)
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_ORG:
        client_kwargs["organization"] = OPENAI_ORG
    if OPENAI_PROJECT:
        client_kwargs["project"] = OPENAI_PROJECT
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    openai_client = OpenAI(**client_kwargs)
except Exception as e:
    print(f"⚠️ Failed to initialize OpenAI client: {e}")
    openai_client = None

# Validate the API key with a lightweight call at startup (clearer early failure)
OPENAI_AVAILABLE = True
try:
    # This will raise if the key is invalid
    _ = openai_client.models.list() if openai_client else None
    masked_key = (OPENAI_API_KEY[:6] + "..." + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else ""
    print(
        "✅ OpenAI client initialized:",
        {
            "model": OPENAI_MODEL,
            "key": masked_key,
            "org": (OPENAI_ORG[:6] + "...") if OPENAI_ORG else None,
            "project": (OPENAI_PROJECT[:6] + "...") if OPENAI_PROJECT else None,
            "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1",
        }
    )
except Exception as e:
    emsg = str(e)
    if "401" in emsg or "invalid_api_key" in emsg.lower() or "Incorrect API key provided" in emsg:
        masked = (OPENAI_API_KEY[:6] + "..." + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else ""
        hint = ""
        if OPENAI_API_KEY.startswith("sk-proj-") and not (OPENAI_PROJECT or OPENAI_ORG):
            hint = " It looks like a project-scoped key; set OPENAI_PROJECT (and optionally OPENAI_ORG)."
        print(f"❌ OpenAI rejected the API key ({masked}). Continuing without LLM.{hint}")
        OPENAI_AVAILABLE = False
    # Non-auth failures should not block app boot; they will be retried on demand
    print(f"⚠️ OpenAI models.list check failed (non-auth): {e}")

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
    # Check if collection exists
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print(f"⚠️ Warning: Collection '{COLLECTION_NAME}' does not exist in Qdrant.")
            print(f"   Run 'upload_to_qdrant.py' first to create and populate the collection.")
    except Exception:
        pass  # Collection check failed, but continue (might be network issue)
except Exception as e:
    raise ValueError(f"❌ Failed to connect to Qdrant: {e}")

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
except Exception as e:
    raise ValueError(f"❌ Failed to load embedding model: {e}")

# NLTK downloads
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ----------------- Helper functions -----------------
def preprocess_with_synonyms(query: str) -> str:
    words = query.split()
    expanded = []
    for word in words:
        try:
            syns = wordnet.synsets(word)
            if syns:
                expanded.append(syns[0].lemmas()[0].name().replace("_", " "))
        except Exception:
            pass
        expanded.append(word)
    return " ".join(expanded)

def classify_query(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["define", "what is", "fees", "duration", "admission", "when"]):
        return "factual"
    if any(kw in q for kw in ["how", "python", "write code", "syntax", "function"]):
        return "technical"
    if any(kw in q for kw in ["compare", "difference", "better", "advantage"]):
        return "complex"
    return "conversational"

# ----------------- Qdrant retrieval -----------------
def retrieve_with_qdrant(query, k=3):
    vec = embedder.encode(query).tolist()
    try:
        res = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=k
        )
        results = [hit.payload.get("text", "") for hit in res if hit.payload.get("text")]
        scores = [hit.score for hit in res if hasattr(hit, "score")]
        # Extract source URLs from chunks
        sources = []
        for hit in res:
            text = hit.payload.get("text", "")
            if text:
                # Extract URL from chunk format: "... [Source: URL]"
                source_match = re.search(r'\[Source:\s*([^\]]+)\]', text)
                if source_match:
                    url = source_match.group(1).strip()
                    sources.append({"url": url, "title": url.split('/')[-1] or url})
        return results, scores, sources
    except Exception as e:
        print(f"⚠️ Qdrant search failed: {e}")
        return ["No Data Available"], [], []

def filter_by_confidence(results, confidences, threshold=0.4):
    if not results:
        return ["No Data Available"]
    if confidences and max(confidences) > threshold:
        return results
    return ["No Data Available"]

def amir_pipeline(query: str):
    q = query.lower().strip()

    # Only treat short greetings as conversation
    if q in ["hi", "hello", "hey", "hii", "good morning", "good evening", "how are you"]:
        return ["__CONVERSATION__"], []

    # Normal pipeline for real questions
    results, confidences, sources = retrieve_with_qdrant(query)
    filtered = filter_by_confidence(results, confidences, threshold=0.4)
    if filtered == ["No Data Available"]:
        expanded = preprocess_with_synonyms(query)
        results, confidences, sources = retrieve_with_qdrant(expanded)
        filtered = filter_by_confidence(results, confidences, threshold=0.35)
    return filtered, sources


# ----------------- OpenAI response -----------------
def _get_openai_client():
    return openai_client

def generate_response_with_openai(query: str, retrieved_chunks):
    # Conversational shortcuts
    if retrieved_chunks == ["__CONVERSATION__"]:
        q = query.lower()
        if "hi" in q or "hello" in q or "hey" in q:
            return "👋 Hi! How can I help you today?"
        if "how are you" in q:
            return "😊 I'm doing great, thanks for asking! How can I assist you?"
        return "🤖 Hello! How can I help you with SPDY Institute info?"

    # If no data found
    if not retrieved_chunks or retrieved_chunks == ["No Data Available"]:
        return "❌ No Data Available"

    # Local extractive fallback (no-LLM): return the most relevant snippet
    def _fallback_answer():
        context_preview = " ".join((retrieved_chunks or [])[:2])
        # Keep it concise
        context_preview = (context_preview[:600] + "…") if len(context_preview) > 600 else context_preview
        return context_preview or "❌ No Data Available"

    # Otherwise → Use OpenAI with retrieved dataset chunks
    # Limit context size to reduce token usage
    limited_chunks = (retrieved_chunks or [])[:3]
    context = "\n".join(limited_chunks)
    if len(context) > 4000:
        context = context[:4000]
    prompt = f"""You are a helpful assistant trained on SPDY Institute information.

Based on the following context, answer the user's question.
If the answer is not present, say 'No Data Available' and do not make anything up.

Context:
{context}

Question:
{query}"""

    # Exponential backoff on 429s
    max_attempts = 5
    backoff = 1.5
    delay = 1.0
    if not OPENAI_AVAILABLE or _get_openai_client() is None:
        return _fallback_answer()
    for attempt in range(1, max_attempts + 1):
        try:
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant trained on SPDY Institute information. Provide accurate, concise answers based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                if attempt == max_attempts:
                    return "⚠️ OpenAI API rate limit exceeded. Please try again shortly."
                # jittered backoff
                import random
                time.sleep(delay + random.uniform(0, 0.5))
                delay *= backoff
                continue
            if "401" in msg or "invalid_api_key" in msg.lower() or "Incorrect API key provided" in msg:
                masked = (OPENAI_API_KEY[:6] + "..." + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else ""
                hint = ""
                if OPENAI_API_KEY.startswith("sk-proj-") and not (OPENAI_PROJECT or OPENAI_ORG):
                    hint = " Set OPENAI_PROJECT (and optionally OPENAI_ORG) if using a project-scoped key."
                # Fall back to extractive answer rather than erroring
                fb = _fallback_answer()
                return fb
            # Surface detailed message for easier debugging (masked key only)
            # Fall back for other errors too
            return _fallback_answer()


# ----------------- ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "⚠️ Please enter a query.", "sources": []})

    # Step 1: Retrieve with AMIR
    context_chunks, sources = amir_pipeline(query)

    # Step 2: Generate response
    answer = generate_response_with_openai(query, context_chunks)

    # Deduplicate sources by URL
    unique_sources = []
    seen_urls = set()
    for source in sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)

    return jsonify({"answer": answer, "sources": unique_sources})

# ----------------- DEBUG ROUTES (local troubleshooting) -----------------
@app.route("/debug/openai", methods=["GET"])
def debug_openai():
    try:
        models = _get_openai_client().models.list()
        first_model = models.data[0].id if getattr(models, "data", None) else None
        return jsonify({
            "ok": True,
            "first_model": first_model,
            "model_count": len(models.data) if getattr(models, "data", None) else 0
        })
    except Exception as e:
        masked = (OPENAI_API_KEY[:6] + "..." + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else ""
        return jsonify({
            "ok": False,
            "error": str(e),
            "key_fingerprint": masked
        }), 500

# ----------------- MAIN -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
