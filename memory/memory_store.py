"""
memory/memory_store.py — Episodic + Semantic memory for the agent.

HUMAN ANALOGY — EPISODIC MEMORY:
Imagine your bank relationship manager. Every time you call, before picking up
they check their notes: "Chandana called last week about a home loan.
She's risk-averse. She prefers WhatsApp updates." They use those notes
to personalize the conversation immediately. That's episodic memory.
Stored facts from past sessions, loaded at the start of each new one.

HUMAN ANALOGY — SEMANTIC MEMORY:
Your brain remembers that "Sachin Tendulkar", "cricket legend", "batting records"
are all about the same thing — even if none of those exact words appear together.
Semantic memory searches by MEANING, not exact keywords.
A vector database does this for AI agents.
"""

import json
import os
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# EPISODIC MEMORY
# Storage: JSON file (swap for PostgreSQL in production)
# ─────────────────────────────────────────────────────────────────────────────

EPISODIC_FILE = "memory/episodic_store.json"


def _load_all_episodic() -> dict:
    os.makedirs("memory", exist_ok=True)
    if not os.path.exists(EPISODIC_FILE):
        return {}
    with open(EPISODIC_FILE) as f:
        return json.load(f)


def _save_all_episodic(data: dict):
    os.makedirs("memory", exist_ok=True)
    with open(EPISODIC_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_episodic_memories(user_id: str, limit: int = 5) -> List[str]:
    """
    Load the most recent memories for a user.
    Called at the START of every session to personalize the experience.
    """
    all_data = _load_all_episodic()
    user_memories = all_data.get(user_id, [])
    # Return most recent N memories
    recent = user_memories[-limit:] if len(user_memories) > limit else user_memories
    return [m["fact"] for m in recent]


def save_episodic_memory(user_id: str, fact: str):
    """
    Save a single fact about a user for future sessions.
    Called at the END of each session after extracting key facts.
    """
    all_data = _load_all_episodic()
    if user_id not in all_data:
        all_data[user_id] = []

    # Avoid storing duplicates
    existing_facts = [m["fact"] for m in all_data[user_id]]
    if fact not in existing_facts:
        all_data[user_id].append({
            "fact": fact,
            "stored_at": datetime.now().isoformat()
        })
        _save_all_episodic(all_data)
        print(f"    [Memory saved]: {fact[:60]}")


def extract_and_save_memories(user_id: str, conversation: List[dict]):
    """
    At end of session: use Claude to extract key facts worth remembering.

    FAILURE SCENARIO this solves:
    Without extraction, you'd store every message — noise and errors included.
    This uses a cheap Claude call to pull out ONLY lasting, useful facts.

    WHAT CAN GO WRONG:
    Claude might extract opinions ("user seemed frustrated") as facts.
    Fix: instruct Claude to extract ONLY objective facts about preferences and context.
    """
    from config import tracked_call, MODELS

    if not conversation:
        return

    convo_text = "\n".join(
        f"{m['role'].upper()}: {m.get('content', '')[:200]}"
        for m in conversation
        if isinstance(m.get("content"), str)
    )

    response, _ = tracked_call(
        model=MODELS["retrieve"],
        messages=[{
            "role": "user",
            "content": f"""Extract 2-3 objective facts about the user from this conversation.
Only facts that would be useful in future sessions — preferences, context, goals.
Do NOT extract emotions, complaints, or one-time questions.
Format: one fact per line, starting with "User "

Conversation:
{convo_text}"""
        }],
        max_tokens=150
    )

    facts = [
        line.strip() for line in response.content[0].text.strip().split("\n")
        if line.strip().startswith("User ")
    ]
    for fact in facts:
        save_episodic_memory(user_id, fact)


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC MEMORY (Vector Store)
# Storage: In-memory for development (swap for ChromaDB/pgvector in production)
# ─────────────────────────────────────────────────────────────────────────────

# In-memory vector store — a list of {"text", "embedding", "metadata"}
# In production: replace with ChromaDB or pgvector
_VECTOR_STORE: List[Dict] = []


def _embed(text: str) -> List[float]:
    """
    Create a vector representation of text.

    PRODUCTION: Use sentence-transformers (free, local):
        pip install sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()

    Or Voyage AI (Anthropic's recommended embedding partner):
        pip install voyageai
        import voyageai
        vo = voyageai.Client()
        result = vo.embed([text], model="voyage-3-lite")
        return result.embeddings[0]

    For this demo: keyword frequency vector (no external dependencies needed)
    """
    keywords = [
        "stock", "invest", "portfolio", "market", "price", "return", "risk",
        "movie", "show", "music", "entertainment", "recommend", "watch",
        "order", "buy", "purchase", "delivery", "track", "cancel",
        "calculate", "sum", "total", "percentage", "tax", "interest",
        "account", "balance", "transfer", "payment", "bank"
    ]
    words = text.lower().split()
    return [float(words.count(k)) for k in keywords]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Measures how similar two vectors are.
    1.0 = identical meaning. 0.0 = completely unrelated.

    HUMAN ANALOGY:
    Two arrows pointing in the same direction = similar (score near 1.0)
    Two arrows pointing opposite directions = dissimilar (score near 0.0)
    Cosine similarity measures the ANGLE between two meaning-arrows.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def store_in_vector_db(text: str, metadata: dict = {}):
    """
    Store a piece of knowledge in the semantic vector store.
    Called when ingesting new documents, FAQs, product info, etc.
    """
    embedding = _embed(text)
    _VECTOR_STORE.append({
        "text": text,
        "embedding": embedding,
        "metadata": {**metadata, "stored_at": datetime.now().isoformat()}
    })


def retrieve_from_vector_db(query: str, top_k: int = 3,
                             min_score: float = 0.05) -> List[Tuple[str, float]]:
    """
    Find the most semantically similar documents to a query.
    Returns: list of (text, similarity_score) tuples.

    FAILURE SCENARIO — LOW SCORES:
    If all similarity scores are below min_score, nothing is returned.
    This is CORRECT behavior — better to say "I don't know" than to hallucinate.
    Do NOT lower min_score to make it return more results.
    That just makes the context noisier and increases hallucination risk.
    """
    if not _VECTOR_STORE:
        return []

    query_emb = _embed(query)
    scored = [
        (doc["text"], _cosine_similarity(query_emb, doc["embedding"]))
        for doc in _VECTOR_STORE
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(text, score) for text, score in scored[:top_k] if score >= min_score]


def seed_knowledge_base():
    """
    Pre-load the vector DB with domain knowledge.

    HUMAN ANALOGY:
    This is like stocking a library before it opens. You put all the reference
    books on the shelves. When a user asks a question, the agent searches
    the library instead of making things up.

    In production: this runs during deployment, not on every request.
    Documents come from PDFs, databases, wikis, etc.
    """
    finance_docs = [
        "A SIP (Systematic Investment Plan) allows investing fixed amounts monthly in mutual funds. Good for rupee cost averaging.",
        "NIFTY 50 is an index of the top 50 companies on NSE India. It represents large-cap Indian stocks.",
        "Fixed deposits in India typically offer 6-7% annual interest for 1-3 year tenures.",
        "ELSS (Equity Linked Savings Scheme) mutual funds offer tax deduction under Section 80C up to Rs 1.5 lakh.",
        "PPF (Public Provident Fund) offers 7.1% tax-free returns with 15-year lock-in. Ideal for risk-averse investors.",
        "Diversification means spreading investments across asset classes to reduce risk. Don't put all eggs in one basket.",
        "P/E ratio (Price to Earnings) measures how much investors pay per rupee of earnings. High P/E may mean overvalued.",
        "SGB (Sovereign Gold Bond) lets you invest in gold digitally. Earns 2.5% annual interest plus gold price gains.",
    ]

    entertainment_docs = [
        "Netflix India popular shows include Scam 1992, Delhi Crime, Sacred Games, and Panchayat.",
        "Disney+ Hotstar streams IPL cricket, Star Wars, Marvel content, and original Indian shows.",
        "Prime Video India has Mirzapur, The Family Man, Patal Lok, and international content.",
        "For Bollywood fans: RRR, KGF Chapter 2, Brahmastra were blockbusters in recent years.",
        "SonyLIV has The Kapil Sharma Show, Taarak Mehta, and Indian Premier League highlights.",
        "If you like thrillers, recommended: Breathe, Special Ops, Tandav, and Aarya on OTT platforms.",
        "For comedy shows: Panchayat (Prime), Kota Factory (Netflix), TVF Pitchers (YouTube) are highly rated.",
        "Hollywood blockbusters on Indian OTT: Avengers series, Fast & Furious, Jurassic Park franchise.",
    ]

    orders_docs = [
        "Order status PROCESSING means payment confirmed, warehouse picking items.",
        "Order status SHIPPED means items dispatched, tracking number available.",
        "Order status OUT_FOR_DELIVERY means delivery agent has the package today.",
        "Standard delivery takes 3-5 business days. Express delivery takes 1-2 days at Rs 299 extra.",
        "Return window is 30 days from delivery. Electronics have 10-day replacement policy.",
        "Cancellation is free before shipping. Post-shipping cancellations incur 5% restocking fee.",
        "EMI available on orders above Rs 5000. 0% interest EMI for 3 months on select cards.",
        "Bulk orders of 10+ items get 15% discount. Corporate accounts get dedicated account manager.",
    ]

    all_docs = finance_docs + entertainment_docs + orders_docs
    for doc in all_docs:
        category = "finance" if doc in finance_docs else ("entertainment" if doc in entertainment_docs else "orders")
        store_in_vector_db(doc, {"category": category})

    print(f"    [VectorDB seeded with {len(all_docs)} documents]")
