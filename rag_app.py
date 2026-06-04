"""
RAG Application — ChromaDB + Google Gemini 2.5 Flash Lite
=========================================================
Topic: Python Data Structures Best Practices

Flow:
  User Question
       ↓
  Retrieve from 3 ChromaDB collections (docs, forums, blogs)
       ↓
  Rerank results by relevance score
       ↓
  Detect contradictions between sources
       ↓
  Ask Gemini to generate a final answer
       ↓
  Print answer + log which sources were used
"""

import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from dotenv import load_dotenv


# ============================================================
# STEP 1 — LOGGING SETUP
# Logs go to both the terminal AND a file called rag.log
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag.log"),   # saves logs to a file
        logging.StreamHandler()           # also prints logs to terminal
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# STEP 2 — LOAD GEMINI API KEY FROM .env FILE
# ============================================================

load_dotenv()  # reads the .env file in the current folder
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    raise ValueError(
        "\n\nGEMINI_API_KEY is missing!\n"
        "1. Copy .env.example  →  rename it to  .env\n"
        "2. Open .env and replace 'your_api_key_here' with your real key\n"
        "   Get a free key at: https://aistudio.google.com/api-keys\n"
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash-lite"
logger.info("Gemini client loaded successfully.")


# ============================================================
# STEP 3 — CHROMADB SETUP
# We create 3 separate collections — one per data source.
# The default embedding function converts text → numbers
# so ChromaDB can find similar chunks later.
# ============================================================

chroma_client = chromadb.Client()
embed_fn = embedding_functions.DefaultEmbeddingFunction()

doc_col   = chroma_client.get_or_create_collection("documentation", embedding_function=embed_fn)
forum_col = chroma_client.get_or_create_collection("forums",        embedding_function=embed_fn)
blog_col  = chroma_client.get_or_create_collection("blogs",         embedding_function=embed_fn)


# ============================================================
# STEP 4 — CHUNKING STRATEGIES
# Different data types need different ways of splitting text.
# ============================================================

def chunk_documentation(text, chunk_size=300, overlap=50):
    """
    Fixed-size chunks with overlap.
    Good for structured reference docs — keeps context across chunk boundaries.
    overlap: the last 50 characters of one chunk repeat at the start of the next.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap   # slide window forward (with overlap)
    return chunks


def chunk_forum_posts(text):
    """
    Each Q&A pair becomes one chunk.
    Forum posts in forum_posts.txt are separated by '---'.
    Keeping Q+A together preserves the conversational context.
    """
    posts = [p.strip() for p in text.split("---") if p.strip()]
    return posts


def chunk_blog_posts(text):
    """
    Split by paragraph (blank line = paragraph break).
    Blog posts flow naturally in paragraphs, so each paragraph is a chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


# ============================================================
# STEP 5 — LOAD DATA FILES AND INDEX INTO CHROMADB
# This runs once. If already indexed, it skips.
# ============================================================

def load_and_index_data():
    """Read the 3 data files, chunk them, and store in ChromaDB."""

    # If already indexed from a previous run, skip to save time
    if doc_col.count() > 0 and forum_col.count() > 0 and blog_col.count() > 0:
        logger.info("Data already indexed. Skipping indexing step.")
        return

    logger.info("Starting data indexing into ChromaDB...")

    # -- Documentation --
    with open("data/documentation.txt", "r") as f:
        doc_text = f.read()
    doc_chunks = chunk_documentation(doc_text)
    doc_col.add(
        documents=doc_chunks,
        ids=[f"doc_{i}" for i in range(len(doc_chunks))],
        metadatas=[{"source": "documentation"} for _ in doc_chunks]
    )
    logger.info(f"Indexed {len(doc_chunks)} documentation chunks.")

    # -- Forum Posts --
    with open("data/forum_posts.txt", "r") as f:
        forum_text = f.read()
    forum_chunks = chunk_forum_posts(forum_text)
    forum_col.add(
        documents=forum_chunks,
        ids=[f"forum_{i}" for i in range(len(forum_chunks))],
        metadatas=[{"source": "forum"} for _ in forum_chunks]
    )
    logger.info(f"Indexed {len(forum_chunks)} forum post chunks.")

    # -- Blog Posts --
    with open("data/blog_posts.txt", "r") as f:
        blog_text = f.read()
    blog_chunks = chunk_blog_posts(blog_text)
    blog_col.add(
        documents=blog_chunks,
        ids=[f"blog_{i}" for i in range(len(blog_chunks))],
        metadatas=[{"source": "blog"} for _ in blog_chunks]
    )
    logger.info(f"Indexed {len(blog_chunks)} blog post chunks.")

    logger.info("All data indexed successfully.")


# ============================================================
# STEP 6 — WEIGHTED RETRIEVAL
# We query all 3 collections and score each result.
# Documentation is most authoritative → highest weight.
# ============================================================

# How much to trust each source (higher = more trusted)
SOURCE_WEIGHTS = {
    "documentation": 1.0,
    "forum":         0.7,
    "blog":          0.6,
}


def retrieve(query, n_results=3):
    """
    Query each ChromaDB collection and return a combined list of results.
    Each result gets a weighted_score = similarity × source_weight.
    """
    logger.info(f"Retrieving chunks for query: '{query}'")
    all_results = []

    for collection, source_name in [
        (doc_col,   "documentation"),
        (forum_col, "forum"),
        (blog_col,  "blog"),
    ]:
        count = collection.count()
        if count == 0:
            continue

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count)
        )

        docs      = results["documents"][0]
        distances = results["distances"][0]  # lower distance = more similar

        for doc, dist in zip(docs, distances):
            # Convert distance → similarity (0 to 1 range)
            similarity = 1 / (1 + dist)
            weighted_score = similarity * SOURCE_WEIGHTS[source_name]

            all_results.append({
                "text":           doc,
                "source":         source_name,
                "raw_score":      similarity,
                "weighted_score": weighted_score,
                "final_score":    weighted_score,  # will be updated during rerank
            })

        logger.info(f"  Retrieved {len(docs)} chunks from '{source_name}'")

    return all_results


# ============================================================
# STEP 7 — RERANKING
# After retrieval, we boost chunks that contain the actual
# words from the query (keyword overlap).
# ============================================================

def rerank(results, query, top_k=5):
    """
    Combine the weighted vector score with keyword overlap score.
    Final score = 70% vector similarity + 30% keyword match.
    Returns the top_k best chunks.
    """
    query_words = set(query.lower().split())

    for result in results:
        text_words = set(result["text"].lower().split())

        # What fraction of query words appear in this chunk?
        if query_words:
            keyword_overlap = len(query_words & text_words) / len(query_words)
        else:
            keyword_overlap = 0

        # Blend the two scores
        result["final_score"] = (result["weighted_score"] * 0.7) + (keyword_overlap * 0.3)

    # Sort: highest final_score first
    reranked = sorted(results, key=lambda x: x["final_score"], reverse=True)

    if reranked:
        logger.info(f"Reranking done. Top chunk source: '{reranked[0]['source']}' "
                    f"(score: {reranked[0]['final_score']:.3f})")

    return reranked[:top_k]


# ============================================================
# STEP 8 — CONTRADICTION DETECTION
# Check if different sources use opposing language.
# If so, we tell Gemini to acknowledge the conflict.
# ============================================================

CONTRADICTION_PAIRS = [
    ("always",      "never"),
    ("fast",        "slow"),
    ("recommended", "avoid"),
    ("prefer",      "don't"),
    ("use",         "avoid"),
    ("efficient",   "inefficient"),
    ("never",       "fine"),
]


def detect_contradictions(results):
    """
    Scan retrieved chunks for opposing keywords across different sources.
    Returns a list of contradiction descriptions (strings).
    """
    # Group text by source
    source_texts = {}
    for r in results:
        source_texts.setdefault(r["source"], []).append(r["text"].lower())

    contradictions_found = []

    for word_a, word_b in CONTRADICTION_PAIRS:
        sources_with_a = [
            s for s, texts in source_texts.items()
            if any(word_a in t for t in texts)
        ]
        sources_with_b = [
            s for s, texts in source_texts.items()
            if any(word_b in t for t in texts)
        ]

        # Only flag if DIFFERENT sources hold opposing views
        if sources_with_a and sources_with_b and set(sources_with_a) != set(sources_with_b):
            contradictions_found.append(
                f"'{word_a}' appears in {sources_with_a}  ←→  "
                f"'{word_b}' appears in {sources_with_b}"
            )

    if contradictions_found:
        logger.warning(f"Contradictions detected: {contradictions_found}")
    else:
        logger.info("No contradictions detected between sources.")

    return contradictions_found


# ============================================================
# STEP 9 — GENERATE ANSWER WITH GEMINI
# We build a prompt that includes the retrieved context,
# a note about contradictions, and the user's question.
# ============================================================

def generate_answer(query, top_chunks, contradictions):
    """Send context + question to Gemini and return its answer."""

    # Build the context block from top retrieved chunks
    context_text = ""
    for i, chunk in enumerate(top_chunks, start=1):
        context_text += (
            f"\n[Source {i} — {chunk['source'].upper()}]"
            f" (relevance score: {chunk['final_score']:.2f})\n"
            f"{chunk['text']}\n"
        )

    # Build contradiction warning (if any)
    contradiction_note = ""
    if contradictions:
        lines = "\n".join(f"  • {c}" for c in contradictions)
        contradiction_note = (
            f"\nNOTE: The sources contain contradictory information:\n{lines}\n"
            f"Please acknowledge these contradictions in your answer and present "
            f"multiple perspectives where relevant.\n"
        )

    # Final prompt
    prompt = f"""You are a helpful technical assistant.
Answer the question using ONLY the information provided in the context below.
If the context does not have enough information, clearly say so.

CONTEXT:
{context_text}
{contradiction_note}
QUESTION: {query}

ANSWER:"""

    logger.info("Sending prompt to Gemini API...")
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    logger.info("Received response from Gemini.")
    return response.text


# ============================================================
# STEP 10 — FULL RAG PIPELINE
# This ties everything together.
# ============================================================

def ask(query):
    """
    Given a user question, run the full RAG pipeline and print the answer.
    """
    print(f"\n{'='*60}")
    print(f" QUESTION: {query}")
    print('='*60)

    # 1. Retrieve relevant chunks from all sources
    raw_results = retrieve(query)

    if not raw_results:
        print("No relevant chunks found. Please check your data files.")
        return

    # 2. Rerank to find the best chunks
    top_chunks = rerank(raw_results, query)

    # 3. Detect contradictions between sources
    contradictions = detect_contradictions(top_chunks)

    # 4. Log source breakdown
    source_counts = {}
    for r in top_chunks:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    logger.info(f"Sources used in final answer: {source_counts}")

    # 5. Generate answer with Gemini
    answer = generate_answer(query, top_chunks, contradictions)

    # 6. Display results
    print(f"\n ANSWER:\n{answer}")
    print(f"\n SOURCES USED: {source_counts}")

    if contradictions:
        print(f" CONTRADICTIONS DETECTED ({len(contradictions)}):")
        for c in contradictions:
            print(f"   • {c}")

    print('='*60)
    return answer


# ============================================================
# ENTRY POINT — Run this file directly to start the chatbot
# ============================================================

if __name__ == "__main__":
    # Index data (only runs once; skips if already indexed)
    load_and_index_data()

    print("\n" + "="*60)
    print("  RAG Chatbot is ready!")
    print("  Topic: Python Data Structures Best Practices")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*60)

    # Simple question loop
    while True:
        try:
            user_input = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        ask(user_input)
