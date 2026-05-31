"""
config.py — Central configuration for OpsMind AI.

All settings live here. Every other module imports from this file.
Changing a setting here propagates everywhere automatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Reads your .env file into environment variables

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent          # The opsmind_ai/ folder
DATA_DIR = ROOT_DIR / "data"              # Where logs and runbooks live
CHROMA_PERSIST_DIR = ROOT_DIR / "chroma_db"  # ChromaDB stores its data here

# ── Embedding model ────────────────────────────────────────────────────────────
# Runs locally on CPU. Downloads ~90MB on first use, then cached.
# This same model is used for BOTH ingestion and retrieval — they must match.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking settings ──────────────────────────────────────────────────────────
# Why 600 chars? Small enough that the LLM gets focused context, big enough
# to contain a full log event or a complete runbook step.
# Why 60 overlap? Ensures no critical sentence gets cut at a boundary.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "opsmind"   # Name of the collection inside ChromaDB

# ── LLM (Groq) ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Best free model on Groq as of 2025

# How many retrieved chunks to send to the LLM per query
TOP_K_RESULTS = 5