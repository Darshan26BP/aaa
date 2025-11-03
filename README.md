# SPDY Institute AI Chatbot 🤖

An AI-powered RAG (Retrieval-Augmented Generation) chatbot for SPDY Institute that delivers accurate responses from institutional data using vector search and NLP.

## 🚀 Features

- **RAG Architecture**: Combines vector search with OpenAI GPT for accurate, contextual responses
- **Vector Database**: Uses Qdrant for efficient similarity search
- **Web Scraping**: Automatically extracts content from SPDY Institute website
- **Smart Chunking**: Intelligent text segmentation with overlap for better context
- **Source Attribution**: Displays source URLs for all responses
- **Responsive UI**: Modern, clean chat interface

## 📋 Prerequisites

- Python 3.8+
- Qdrant Cloud account (or self-hosted Qdrant instance)
- OpenAI API key
- Playwright (for web scraping with JavaScript rendering)

## 🛠️ Installation

1. **Clone/Download the repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers (for web scraping):**
   ```bash
   playwright install chromium
   ```

4. **Configure environment variables:**
   
   Update `Botpro/.env` with your credentials:
   ```env
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   QDRANT_COLLECTION=spdy-chatbot
   OPENAI_API_KEY=your_openai_api_key_here
   CHUNKS_FILE=spdy_website_chunks.json
   ```

## 🔄 Setup Pipeline

### Step 1: Extract Website Content

Extract content from SPDY Institute website:

```bash
cd Botpro
python extract_spdy_website_text.py
```

This creates:
- `spdy_website_pages_full.json` - Structured page data
- `spdy_website_full_content.txt` - Flattened text content

### Step 2: Chunk the Content

Split the extracted content into searchable chunks:

```bash
python chunk_spdy.py
```

This creates:
- `spdy_website_chunks.json` - Array of text chunks ready for embedding

### Step 3: Upload to Qdrant

Generate embeddings and upload chunks to Qdrant vector database:

```bash
python upload_to_qdrant.py
```

This will:
- Create embeddings using `all-MiniLM-L6-v2` model
- Upload vectors to Qdrant collection
- Store chunk text as metadata

## 🎯 Running the Chatbot

Start the Flask application:

```bash
python app.py
```

The chatbot will be available at: `http://localhost:5000`

Open your browser and start chatting!

## 📁 Project Structure

```
Botpro/
├── app.py                          # Main Flask application
├── extract_spdy_website_text.py     # Web scraper
├── chunk_spdy.py                   # Text chunking script
├── upload_to_qdrant.py             # Qdrant upload script
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── templates/
│   └── index.html                  # Frontend UI
└── static/
    └── b4.jpg                      # Background image
```

## 🔧 Configuration

### Adjusting Chunking Parameters

Edit `chunk_spdy.py`:
- `MAX_WORDS = 120` - Maximum words per chunk
- `OVERLAP = 30` - Word overlap between chunks

### Changing OpenAI Model

Edit `app.py`, line 135:
```python
model="gpt-3.5-turbo"  # Change to "gpt-4" for better quality
```

### Adjusting Retrieval Settings

Edit `app.py`:
- `retrieve_with_qdrant(query, k=5)` - Number of chunks to retrieve
- `threshold=0.4` - Confidence threshold for filtering results

## 🔄 Updating the Knowledge Base

To refresh the chatbot with new content:

1. Run `extract_spdy_website_text.py` again
2. Run `chunk_spdy.py` again
3. Run `upload_to_qdrant.py` (it will update existing collection)

## 🐛 Troubleshooting

**Qdrant connection errors:**
- Verify your QDRANT_URL and QDRANT_API_KEY in `.env`
- Check if your Qdrant instance is accessible

**OpenAI API errors:**
- Verify your OPENAI_API_KEY is valid
- Check your OpenAI account quota/limits

**Empty responses:**
- Ensure chunks have been uploaded to Qdrant
- Check if the collection name matches in `.env`

**Web scraping issues:**
- Make sure Playwright is installed: `playwright install chromium`
- Verify the target website URL is accessible

## 📝 Notes

- The first run of `extract_spdy_website_text.py` may take a while as it crawls the entire website
- Large websites may hit rate limits; adjust `MAX_PAGES` if needed
- Chunking preserves source URLs for proper attribution

## 🤝 Contributing

Feel free to submit issues or pull requests!

## 📄 License

This project is for SPDY Institute internal use.

