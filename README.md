# Research Assistant — RAG-based Document Q&A

A project built to understand how Retrieval Augmented Generation (RAG) works from scratch.
You upload your own documents, ask questions in plain English, and get answers grounded in what your documents actually say — not general internet knowledge.

---

## Why I built this

Honestly, just to learn.

RAG is one of those terms that gets thrown around a lot in AI — but most tutorials either skip the internals entirely ("just use LangChain") or go too deep into math. I wanted to build it from scratch using the actual components: a real embedding model, a real vector store, a real LLM — so I could see exactly what happens at each step and why it works.

This project is the result of that. It's not production-ready. It's a learning project, and it's meant to be read that way.

---

## What it does

- Upload PDFs, text files, scrape web URLs, or pull Wikipedia articles
- Ask questions about your documents in plain English
- Get answers that cite which document and chunk they came from
- Hold a multi-turn conversation — the assistant remembers what it said earlier
- Runs entirely in Google Colab, no local setup needed

---

## Architecture

The system works in two phases.

```
PHASE 1 — Indexing (run once per document set)
─────────────────────────────────────────────
Your documents (PDF / TXT / URL / Wikipedia)
        │
        ▼
   Text extraction
        │
        ▼
   Chunking (1000-char pieces with 150-char overlap)
        │
        ▼
   Embedding model: all-MiniLM-L6-v2
   converts each chunk → 384-dimensional vector
        │
        ▼
   FAISS index saved to disk
   (stores all vectors + original chunk text)


PHASE 2 — Querying (every time you ask a question)
───────────────────────────────────────────────────
Your question (plain text)
        │
        ▼
   Same embedding model → question vector
        │
        ▼
   FAISS similarity search → top 6 nearest chunks
        │
        ▼
   Prompt builder:
   "Here is context: [chunk1][chunk2]... Answer this: [question]"
        │
        ▼
   Groq API → llama-3.1-8b-instant
        │
        ▼
   Answer (grounded in your documents) + source citations
```

**Key insight:** the embedding model puts similar meanings geometrically close together in vector space. So "what methodology was used?" and a chunk saying "we evaluated using a confusion matrix..." will be close — even if they share no exact words. That's what makes the retrieval meaningful.

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Notebook environment | Google Colab | No local setup, free GPU |
| Document parsing | PyPDF2, BeautifulSoup, wikipedia-api | Handles all source types |
| Embedding model | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, no API key |
| Vector store | FAISS (faiss-cpu) | Fast similarity search, saves to disk |
| LLM | Groq — llama-3.1-8b-instant | Free tier, fast inference |
| UI | ipywidgets | Text box inside Colab, no frontend needed |

---

## How to run it

1. Open the notebook in Google Colab
2. Add your Groq API key to Colab Secrets (🔑 icon → name it `GROQ_API_KEY`)
3. Run Cell 1 — installs dependencies
4. Run Cell 2 — loads config and clients
5. Run Cells 3 and 4 — sets up loaders and chunker
6. Run the **clear cell** if switching documents (wipes old index)
7. Run Cell 5 — upload your documents and build the index
8. Run Cell 6 — loads the index into memory
9. Run Cells 7 and 8 — retriever and generator
10. Run Cell 9 — opens the chat UI, start asking questions

If your Colab runtime restarts, re-run Cell 1, Cell 2, and Cell 6 (load index) — you don't need to re-embed unless you want new documents.

---

## Current limitations

**Chunk quality is imperfect**
Splitting by character count is a blunt instrument. A 1000-character chunk might cut a sentence in half, or group two unrelated paragraphs together. Better approaches (splitting by sentence or paragraph boundaries) would improve retrieval accuracy.

**No persistent storage**
The FAISS index lives in Colab's temporary filesystem. When the runtime disconnects, the index is gone and you have to re-embed your documents. Mounting Google Drive would fix this.

**Retrieval doesn't always find the right chunks**
If your question uses different words than the document does (synonyms, different phrasing), the embedding distance can be large even when the content is relevant. A re-ranking step using a cross-encoder would help here.

**Multi-document confusion**
If you upload documents on completely different topics, the retriever sometimes pulls chunks from the wrong document because it's optimizing for vector similarity, not topic relevance. The current fix is to clear and re-index whenever you switch topics.

**No streaming**
The answer only appears after the full Groq response is ready. For long answers this feels slow. The Groq API supports streaming — it just isn't implemented here.

**Context window has a ceiling**
We inject 6 chunks into the prompt. For very long answers that need more context than 6 chunks can provide, the answer will be incomplete. Increasing TOP_K helps but also increases the chance of irrelevant chunks getting in.

---

## What I learned building this

- RAG is not one thing — it's a pipeline of four separate concerns: loading, chunking, retrieval, and generation. Each one has its own failure modes.
- The embedding model is doing the heavy lifting for retrieval quality. The LLM is just the final formatting step.
- Chunk size matters more than it sounds. Too small and you lose context. Too large and you retrieve too much noise.
- Conversation history and retrieval context are separate things. History tells the model what it said before. Context tells it what the documents say now. Mixing them up causes hallucinations.
- "The model said something wrong" is often actually "the retriever pulled the wrong chunks."

---

## Possible next steps

- Sentence-boundary chunking instead of character-count chunking
- Google Drive integration for persistent index storage
- Cross-encoder re-ranking for better retrieval
- Streaming responses
- Support for more source types (DOCX, CSV, YouTube transcripts)
