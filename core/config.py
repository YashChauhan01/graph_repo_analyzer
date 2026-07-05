"""
core/config.py
──────────────
Central configuration loaded once from .env.
All other modules import from here — no scattered os.getenv() calls.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama3-70b-8192"))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))

    # ChromaDB
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "code_embeddings"))

    # Embedding
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "local"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Ingestion
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "60")))
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    backoff_max_tries: int = field(default_factory=lambda: int(os.getenv("BACKOFF_MAX_TRIES", "5")))

    def validate(self) -> None:
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example → .env and fill in the values."
            )


# Single shared instance imported across all modules
cfg = Config()
