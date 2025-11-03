# chunk_spdy.py
import json
from pathlib import Path

INPUT_FILE = "spdy_website_pages_full.json"  # extractor output
OUTPUT_FILE = "spdy_website_chunks.json"

# chunking params
MAX_WORDS = 120
OVERLAP = 30

def split_into_chunks(text, max_words=MAX_WORDS, overlap=OVERLAP):
    """Split long text into chunks with word overlap."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

def normalize_text(s: str) -> str:
    """Normalize whitespace and trim."""
    return " ".join(s.split()).strip()

def make_heading_chunk(tag: str, text: str) -> str:
    t = normalize_text(text)
    if not t:
        return ""
    return f"[{tag.upper()}] {t}"

def process_page(page: dict):
    """
    Build chunks from:
      - page title
      - page headings (structured)
      - page.sections (each has heading, text, tag)
      - fallback to page['text']
    Deduplicate similar chunks.
    """
    chunks = []
    seen = set()  # normalized text set to avoid duplicates

    def add_chunk(text):
        if not text:
            return
        n = normalize_text(text)
        if not n:
            return
        key = n.lower()
        if key in seen:
            return
        seen.add(key)
        chunks.append(n)

    # Title
    title = page.get("title") or ""
    if title:
        add_chunk(make_heading_chunk("TITLE", title))

    # Top-level headings (if extractor saved them)
    for h in page.get("headings", []):
        if isinstance(h, dict):
            tag = h.get("tag", "h")
            text = h.get("text", "")
            add_chunk(make_heading_chunk(tag, text))
        else:
            add_chunk(make_heading_chunk("H", str(h)))

    # Sections (preferred source)
    sections = page.get("sections", [])
    for sec in sections:
        # sec may be dict {"heading":..., "text":..., "tag":...}
        sec_heading = sec.get("heading") if isinstance(sec, dict) else None
        sec_text = sec.get("text") if isinstance(sec, dict) else (str(sec) if sec else "")
        sec_tag = sec.get("tag") if isinstance(sec, dict) else ""
        # Build a combined chunk that preserves heading context
        if sec_heading and sec_heading != "General":
            combined = f"[{sec_tag.upper() if sec_tag else 'SEC'}] {sec_heading} — {sec_text}"
        else:
            combined = f"[{sec_tag.upper() if sec_tag else 'SEC'}] {sec_text}"
        # Add smaller sub-chunks if section is long
        for c in split_into_chunks(combined):
            add_chunk(c)

    # Page-level text fallback (in case sections were empty)
    page_text = page.get("text") or page.get("content") or page.get("body") or ""
    if page_text:
        # Split page_text into chunks and add, but avoid adding if already covered
        for c in split_into_chunks(page_text):
            add_chunk(c)

    return chunks

def main():
    print("📑 Loading extracted data...")
    if not Path(INPUT_FILE).exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    data = json.loads(Path(INPUT_FILE).read_text(encoding="utf-8"))

    all_chunks = []
    for page in data:
        page_chunks = process_page(page)
        # optionally prefix each chunk with URL metadata (kept minimal so embeddings focus on text)
        url = page.get("final_url") or page.get("url") or ""
        if url:
            # include url as a trailing marker for traceability (not huge)
            page_chunks = [f"{c}\n\n[Source: {url}]" for c in page_chunks]
        all_chunks.extend(page_chunks)

    # Save chunks as JSON array of strings
    Path(OUTPUT_FILE).write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Total chunks created: {len(all_chunks)} -> saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
