import os
import re
import json
import time
import hashlib
import tempfile
import traceback
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ---------------- CONFIG ----------------
START_URL = "https://spyd.vercel.app/"   # change to https://spdy.ac.in/ when ready
MAX_DEPTH = 3
MAX_PAGES = 500
OUTPUT_JSON = "spdy_website_pages_full.json"
OUTPUT_TEXT = "spdy_website_full_content.txt"

USER_AGENT = "Mozilla/5.0 (compatible; SPDYBot/1.0; +https://example.com/bot)"
NAV_TIMEOUT_MS = 60000
SCROLL_PAUSE = 0.5
WAIT_AFTER_RENDER = 1.0
MAX_XHR_BODY_CHARS = 20000

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
session.mount("http://", requests.adapters.HTTPAdapter(max_retries=3))
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))

# ---------------- helpers ----------------
def clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def compute_hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            p = ""
        if isinstance(p, (dict, list)):
            p = json.dumps(p, ensure_ascii=False, sort_keys=True)
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()

def safe_request_get_bytes(url, timeout=20):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def extract_text_from_pdf_bytes(pdf_bytes):
    if not pdf_bytes:
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        text_pages = []
        doc = fitz.open(tmp_path)
        for page in doc:
            txt = page.get_text("text") or ""
            text_pages.append(txt)
        doc.close()
        try: os.unlink(tmp_path)
        except: pass
        return clean("\n".join(text_pages))
    except Exception:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            texts = []
            with pdfplumber.open(tmp_path) as pdf:
                for p in pdf.pages:
                    texts.append(p.extract_text() or "")
            try: os.unlink(tmp_path)
            except: pass
            return clean("\n".join(texts))
        except Exception as e:
            return f"[PDF-ERROR] {e}"

def extract_text_from_image_bytes(img_bytes):
    if not img_bytes:
        return ""
    try:
        from io import BytesIO
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        txt = pytesseract.image_to_string(img)
        return clean(txt)
    except Exception:
        return ""

def table_to_text(bs_table):
    rows = []
    for tr in bs_table.find_all("tr"):
        cells = [clean(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)

CLICK_SELECTORS = [
    '[aria-expanded="false"]', '[data-toggle="collapse"]', '.accordion .accordion-button',
    '.accordion-button', '.collapse-button', '.show-more', '.read-more', '.morelink', '.tab',
    '.nav-link', 'button'
]

def click_reveal_buttons(page):
    for sel in CLICK_SELECTORS:
        try:
            elems = page.query_selector_all(sel)
        except: elems = []
        for e in elems:
            try:
                if not e.is_visible(): continue
                try: e.click(timeout=2000); time.sleep(0.2)
                except: 
                    try: page.evaluate("(el)=>el.click()", e); time.sleep(0.2)
                    except: pass
            except: pass

def parse_html_structured(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    headings, sections = [], []

    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            tx = clean(h.get_text(" ", strip=True))
            if tx:
                headings.append({"tag": f"h{level}", "text": tx})

    paragraphs = [clean(p.get_text(" ", strip=True)) for p in soup.find_all("p") if clean(p.get_text())]
    lists = [[clean(li.get_text(" ", strip=True)) for li in ul.find_all("li")] for ul in soup.find_all(["ul", "ol"])]
    tables = [table_to_text(t) for t in soup.find_all("table")]
    jsonld = []
    for s in soup.find_all("script", type="application/ld+json"):
        try: jsonld.append(json.loads(s.string or "{}"))
        except: jsonld.append({"raw": (s.get_text() or "")[:500]})
    anchors = [a.get("href") for a in soup.find_all("a", href=True)]
    images = [{"src": urljoin(base_url,img.get("src")),"alt":img.get("alt","")} for img in soup.find_all("img") if img.get("src")]

    # create pin-to-pin sections (heading + next paragraph or list or table)
    sections = []
    elems = soup.find_all(["h1","h2","h3","h4","h5","h6","p","ul","ol","table"])
    current_heading = None
    for el in elems:
        tag = el.name.lower()
        text = clean(el.get_text(" ", strip=True))
        if not text: continue
        if tag in ["h1","h2","h3","h4","h5","h6"]:
            current_heading = text
        else:
            sections.append({"heading": current_heading or "General", "text": text, "tag": tag})

    return {
        "headings": headings,
        "paragraphs": paragraphs,
        "lists": lists,
        "tables": tables,
        "jsonld": jsonld,
        "anchors": anchors,
        "images": images,
        "sections": sections
    }

def crawl_site(start_url, max_depth=2, max_pages=300):
    parsed = urlparse(start_url)
    base_domain = parsed.netloc
    visited, results = set(), []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT, ignore_https_errors=True)
        to_visit = [(start_url, 0)]

        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)
            norm = url.rstrip("/")
            if norm in visited or depth > max_depth: continue
            visited.add(norm)

            try:
                if norm.lower().endswith(".pdf"):
                    pdf_bytes = safe_request_get_bytes(norm)
                    pdf_text = extract_text_from_pdf_bytes(pdf_bytes) if pdf_bytes else ""
                    ch = compute_hash(pdf_text)
                    results.append({"url": norm,"final_url": norm,"title":os.path.basename(norm),
                                    "text": pdf_text,"sections":[{"heading":"PDF","text":pdf_text}],
                                    "content_hash": ch})
                    continue

                page = context.new_page()
                captured_xhr = []

                def _on_response(resp):
                    try:
                        if resp.request.resource_type in ("xhr","fetch") or resp.url.endswith(".json"):
                            body = ""
                            try: body = resp.text()[:MAX_XHR_BODY_CHARS]
                            except: pass
                            captured_xhr.append({"url":resp.url,"status":resp.status,"body":clean(body)})
                    except: pass
                page.on("response", _on_response)

                try:
                    page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_MS)
                except PlaywrightTimeoutError:
                    page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)

                click_reveal_buttons(page)

                # scroll to trigger lazy load
                prev_height=-1
                for _ in range(12):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(SCROLL_PAUSE)
                    new_height=page.evaluate("() => document.body.scrollHeight")
                    if new_height==prev_height: break
                    prev_height=new_height

                time.sleep(WAIT_AFTER_RENDER)
                html = page.content()
                dom_text = page.evaluate("() => document.body ? document.body.innerText : ''")
                structured = parse_html_structured(html,url)

                combined_text = "\n".join([s["text"] for s in structured.get("sections",[]) if s.get("text")])

                # PDFs & Images OCR
                pdf_texts = []
                for p in structured.get("anchors",[]) or []:
                    p_abs=urljoin(url,p)
                    if p_abs.lower().endswith(".pdf"):
                        b=safe_request_get_bytes(p_abs)
                        t=extract_text_from_pdf_bytes(b) if b else ""
                        pdf_texts.append({"url":p_abs,"text":t})
                        if t: combined_text += "\n"+t

                images_ocr = []
                for img in structured.get("images",[]):
                    src=img.get("src")
                    if not src: continue
                    b=safe_request_get_bytes(src)
                    if b:
                        t=extract_text_from_image_bytes(b)
                        if t:
                            images_ocr.append({"url":src,"alt":img.get("alt"),"text":t})
                            combined_text += "\n"+t

                content_hash = compute_hash(combined_text,"".join([r.get("body","") for r in captured_xhr]))

                rec = {
                    "url": url,
                    "final_url": page.url,
                    "title": clean(page.title() or ""),
                    "text": clean(combined_text),
                    "sections": structured.get("sections",[]),
                    "pdf_texts": pdf_texts,
                    "images_ocr": images_ocr,
                    "xhr": captured_xhr,
                    "content_hash": content_hash,
                    "html": html
                }
                results.append(rec)

                # queue same-domain anchors
                for a in structured.get("anchors",[]) or []:
                    if not a or a.startswith(("mailto:","tel:","javascript:")): continue
                    try:
                        full=urljoin(url,a)
                        if urlparse(full).netloc==base_domain:
                            norm_full=full.split("#")[0].rstrip("/")
                            if norm_full not in visited:
                                to_visit.append((norm_full, depth+1))
                    except: pass

                page.close()
            except Exception as e:
                print(f"[Error crawling] {url} -> {e}")
                traceback.print_exc()
                try: page.close()
                except: pass
                continue

        try: context.close(); browser.close()
        except: pass

    return results  # ✅ FIX: return collected pages

# ---------------- runner ----------------
if __name__ == "__main__":
    print("Starting crawl:", START_URL)
    pages = crawl_site(START_URL, max_depth=MAX_DEPTH, max_pages=MAX_PAGES)
    print("Crawl finished. Pages collected:", len(pages))

    # Write structured JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    # Write flattened TXT for chunking
    with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"\n--- PAGE: {p.get('final_url') or p.get('url')} ---\n")
            if p.get("title"):
                f.write(f"Title: {p.get('title')}\n")
            f.write((p.get("text") or "") + "\n")
            # PDF texts
            for pdf in p.get("pdf_texts", []):
                f.write(f"[PDF: {pdf.get('url')}]\n{pdf.get('text')}\n")
            # Images OCR
            for ocr in p.get("images_ocr", []):
                f.write(f"[Image OCR: {ocr.get('url')}]\n{ocr.get('text')}\n")

    print("Saved:", OUTPUT_JSON, OUTPUT_TEXT)
