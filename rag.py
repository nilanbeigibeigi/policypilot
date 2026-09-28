"""PolicyPilot — the RAG engine.

Three steps:
  1. RETRIEVE: turn the question into a vector and find the most similar document chunks
  2. AUGMENT:  put those chunks into the prompt as numbered sources
  3. GENERATE: the LLM answers ONLY from those sources and cites them like [1]

CLI:
  python rag.py build                 # build the index
  python rag.py ask "How many sick days do I get?"
"""
import glob
import hashlib
import json
import os
import re
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")
INDEX_DIR = os.path.join(HERE, "index")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
EMBEDDER = os.environ.get("EMBEDDER", "ollama")   # "ollama" or "tfidf" (offline fallback)

NO_ANSWER = "I don't know based on the company documents."

SYSTEM_PROMPT = f"""You are PolicyPilot, an assistant for Saffron Market employees.
Answer the question using ONLY the numbered sources below.
Rules:
- Cite the source number for every fact, like [1] or [2].
- Be short and direct (1-3 sentences).
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Never use outside knowledge and never guess."""


# ---------------------------------------------------------------- 1. load + chunk
def load_chunks(docs_dir: str = DOCS_DIR, max_chars: int = 900) -> list[dict]:
    """Split each markdown doc into chunks by '## ' section, so every chunk has a citation."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        text = open(path, encoding="utf-8").read()
        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else os.path.basename(path)
        for part in re.split(r"^## ", text, flags=re.M)[1:]:
            section, _, body = part.partition("\n")
            body = body.strip()
            # very long sections are split by paragraph
            pieces, buf = [], ""
            for para in body.split("\n\n"):
                if buf and len(buf) + len(para) > max_chars:
                    pieces.append(buf); buf = ""
                buf = (buf + "\n\n" + para).strip()
            if buf:
                pieces.append(buf)
            for p in pieces:
                chunks.append({
                    "doc": os.path.basename(path),
                    "title": title,
                    "section": section.strip(),
                    # title + section go into the embedded text: helps retrieval a lot
                    "text": p,
                })
    return chunks


def _embed_text(c: dict) -> str:
    return f"{c['title']} — {c['section']}\n{c['text']}"


# ---------------------------------------------------------------- 2. embeddings
def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class OllamaEmbedder:
    name = "ollama"

    def fit(self, texts):  # nothing to learn
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        # nomic-embed-text works best with these task prefixes
        out = _post("/api/embed", {"model": EMBED_MODEL, "input": texts})
        return _normalize(np.array(out["embeddings"], dtype=np.float32))

    def encode_docs(self, texts):
        return self.encode([f"search_document: {t}" for t in texts])

    def encode_query(self, q):
        return self.encode([f"search_query: {q}"])[0]


class TfidfEmbedder:
    """Offline fallback: keyword-based vectors. No model download, used by the tests."""
    name = "tfidf"

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(texts)
        return self

    def encode_docs(self, texts):
        return _normalize(self.v.transform(texts).toarray().astype(np.float32))

    def encode_query(self, q):
        return self.encode_docs([q])[0]


# ---------------------------------------------------------------- index
class Index:
    def __init__(self, chunks, vectors, embedder):
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder

    @classmethod
    def build(cls, embedder_name: str = EMBEDDER, docs_dir: str = DOCS_DIR, use_cache: bool = True):
        chunks = load_chunks(docs_dir)
        texts = [_embed_text(c) for c in chunks]
        emb = (TfidfEmbedder() if embedder_name == "tfidf" else OllamaEmbedder()).fit(texts)

        if emb.name == "ollama" and use_cache:
            # cache vectors on disk; rebuild automatically when any document changes
            digest = hashlib.sha256((EMBED_MODEL + "".join(texts)).encode()).hexdigest()
            vec_path = os.path.join(INDEX_DIR, "vectors.npy")
            meta_path = os.path.join(INDEX_DIR, "meta.json")
            if os.path.exists(meta_path) and json.load(open(meta_path)).get("digest") == digest:
                return cls(chunks, np.load(vec_path), emb)
            vectors = emb.encode_docs(texts)
            os.makedirs(INDEX_DIR, exist_ok=True)
            np.save(vec_path, vectors)
            json.dump({"digest": digest, "model": EMBED_MODEL, "chunks": len(chunks)}, open(meta_path, "w"))
            return cls(chunks, vectors, emb)

        return cls(chunks, emb.encode_docs(texts), emb)

    def search(self, question: str, k: int = 3) -> list[dict]:
        """Cosine similarity (vectors are normalized, so it's just a dot product)."""
        q = self.embedder.encode_query(question)
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in top]


# ---------------------------------------------------------------- 3. generate
def build_prompt(question: str, sources: list[dict]) -> str:
    blocks = [f"[{i}] ({s['title']} — {s['section']})\n{s['text']}" for i, s in enumerate(sources, 1)]
    return "SOURCES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"


def generate(question: str, sources: list[dict], llm: str = "ollama") -> str:
    prompt = build_prompt(question, sources)
    if llm == "ollama":
        out = _post("/api/chat", {
            "model": CHAT_MODEL, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}]})
        return out["message"]["content"].strip()
    if llm == "claude":
        import anthropic  # pip install anthropic ; needs ANTHROPIC_API_KEY
        msg = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL, max_tokens=400, temperature=0, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    if llm == "none":  # retrieval only: show the best passage
        s = sources[0]
        return f"(retrieval only) Best match — {s['title']} › {s['section']}:\n\n{s['text']} [1]"
    raise ValueError(f"Unknown llm {llm!r}")


def answer(index: Index, question: str, k: int = 3, llm: str = "ollama") -> dict:
    sources = index.search(question, k)
    text = generate(question, sources, llm)
    return {"question": question, "answer": text, "sources": sources}


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "build":
        idx = Index.build(use_cache=False)
        print(f"Indexed {len(idx.chunks)} chunks with {idx.embedder.name}.")
    elif cmd == "ask" and len(sys.argv) > 2:
        llm = os.environ.get("LLM", "ollama")
        r = answer(Index.build(), " ".join(sys.argv[2:]), llm=llm)
        print("\n" + r["answer"] + "\n")
        for i, s in enumerate(r["sources"], 1):
            print(f"  [{i}] {s['title']} › {s['section']}  (score {s['score']})")
    else:
        print(__doc__)
