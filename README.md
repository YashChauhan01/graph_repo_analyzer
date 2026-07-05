# GraphRAG Code Analyzer

Ask questions about any Java or Python codebase in plain English. Point it at a repo, let it ingest, then ask things like:

- *"What does PaymentService depend on?"*
- *"How does the auth middleware handle expired tokens?"*
- *"Which classes extend BaseRepository?"*

Under the hood it builds a Neo4j graph of your code's structure (classes, methods, call chains) and a ChromaDB vector store of what the code actually does. A LangGraph agent figures out which one to query — or both if your question is ambiguous — and Groq (Llama 3) generates the answer.

---

## Setup

You'll need Neo4j running and a Groq API key ([free here](https://console.groq.com)).

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY and NEO4J_PASSWORD
```

Quickest way to get Neo4j up:

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword neo4j:5
```

---

## Usage

```bash
# ingest a repo
python main.py ingest /path/to/repo

# single question
python main.py query "What does UserService call?"

# interactive shell
python main.py shell
```

---

## Why it's fast on tokens

AST parsing strips getters, setters, and boilerplate before anything hits the vector store. The router means most questions only touch one database. Context gets trimmed to a token budget before the LLM call. Ends up ~70% fewer tokens than dumping raw code into a prompt.

---

## Limitations

Only Java and Python for now. The query router uses keyword heuristics so occasionally misroutes weird questions — hybrid mode catches most of those. ChromaDB is file-based by default, fine for local use but you'd swap it for Pinecone or Weaviate in production.