# app.py
import os
from flask import Flask, request, jsonify, render_template
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import nltk
from nltk.corpus import wordnet

# ----------------- CONFIG -----------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mite-chatbot")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ----------------- Initialize -----------------
app = Flask(__name__)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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
def retrieve_with_qdrant(query, k=5):
    vec = embedder.encode(query).tolist()
    try:
        res = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=k
        )
        results = [hit.payload.get("text", "") for hit in res if hit.payload.get("text")]
        scores = [hit.score for hit in res if hasattr(hit, "score")]
        return results, scores
    except Exception as e:
        print(f"⚠️ Qdrant search failed: {e}")
        return ["No Data Available"], []

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
        return ["__CONVERSATION__"]

    # Normal pipeline for real questions
    results, confidences = retrieve_with_qdrant(query)
    filtered = filter_by_confidence(results, confidences, threshold=0.4)
    if filtered == ["No Data Available"]:
        expanded = preprocess_with_synonyms(query)
        results, confidences = retrieve_with_qdrant(expanded)
        filtered = filter_by_confidence(results, confidences, threshold=0.35)
    return filtered


# ----------------- Gemini response -----------------
def generate_response_with_gemini(query: str, retrieved_chunks):
    # Conversational shortcuts
    if retrieved_chunks == ["__CONVERSATION__"]:
        q = query.lower()
        if "hi" in q or "hello" in q or "hey" in q:
            return "👋 Hi! How can I help you today?"
        if "how are you" in q:
            return "😊 I’m doing great, thanks for asking! How can I assist you?"
        return "🤖 Hello! How can I help you with MITE info?"

    # If no data found
    if not retrieved_chunks or retrieved_chunks == ["No Data Available"]:
        return "❌ No Data Available"

    # Otherwise → Use Gemini with retrieved dataset chunks
    context = "\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful assistant trained on MITE college information.

Based on the following context, answer the user's question.
If the answer is not present, say 'No Data Available' and do not make anything up.

Context:
{context}

Question:
{query}
"""
    try:
        resp = gemini_model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Gemini quota exceeded. Please wait a minute before trying again."
        return f"⚠️ Error contacting LLM: {e}"


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
    context_chunks = amir_pipeline(query)

    # Step 2: Generate response
    answer = generate_response_with_gemini(query, context_chunks)

    # For now, sources are empty
    return jsonify({"answer": answer, "sources": []})

# ----------------- MAIN -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
