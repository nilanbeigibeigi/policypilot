# 🧭 PolicyPilot — RAG assistant for company documents

Employees ask questions in plain English; PolicyPilot answers **only from the company's own documents**, cites the exact section, and says *"I don't know"* instead of guessing.

> **Q:** How many sick days do I get?
> **A:** You get 5 paid sick days per calendar year after 90 days of employment [1].
> 📚 [1] Employee Handbook › Sick Days

Built for a fictional Persian grocery store, **Saffron Market** (6 policy documents: handbook, opening/closing, food safety, returns, loyalty, safety). Runs **100% free and local** with Ollama — no company data leaves the computer.

## Why I built it
As an assistant manager in retail, I answered the same questions from staff every week — refunds, fridge temperatures, sick days. The answers were all in documents nobody read. This is the most common AI project companies start with: *"let our people ask our documents."*

## How it works (RAG)

```
Question
   │  1. RETRIEVE  embed the question → cosine similarity → top-3 document chunks
   ▼
Top-3 sections (with doc + section names)
   │  2. AUGMENT   put them in the prompt as numbered sources [1] [2] [3]
   ▼
LLM (Llama 3.2 local, or Claude)
   │  3. GENERATE  answer ONLY from the sources, cite them, or refuse
   ▼
Answer + citations
```

Design choices:
- **Chunk by section heading** → every answer can cite *document › section*.
- **Title + section are embedded with the text** → better search.
- **Strict prompt + temperature 0** → no outside knowledge, fixed refusal sentence.
- **Vector cache** → embeddings are only recomputed when a document changes.
- **Evaluation set** → accuracy is measured, not assumed.

## Evaluation

`eval_questions.json` has 23 questions: 19 answerable (with expected section + expected fact) and 4 **unanswerable** ones (WiFi password, manager salary…) that must be refused.

| Metric | What it checks |
|---|---|
| Retrieval hit rate | Correct section is in the top-k sources |
| Answer accuracy | Answer contains the expected fact |
| Refusal accuracy | Unanswerable questions get "I don't know" |

Keyword search (TF-IDF) baseline: **89.5% at top-1 → 100% at top-3**, which is why 3 sources are sent to the model. Run `python eval.py` to produce your own report in `reports/`.

**Result (Llama 3.2 3B + nomic-embed-text, local):** 100% retrieval · 100% answer accuracy · 100% refusal accuracy on 23 questions (~9 s per answer on a laptop CPU).

## Setup

1. Install [Ollama](https://ollama.com), then:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
2. Install Python packages and test:
   ```bash
   pip install -r requirements.txt
   pytest -q
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Evaluate:
   ```bash
   python eval.py
   ```

**Options** (environment variables): `LLM=claude` (needs `pip install anthropic` + `ANTHROPIC_API_KEY`), `LLM=none` (retrieval only), `EMBEDDER=tfidf` (no Ollama needed for search), `TOP_K=3`.

**Use your own documents:** drop `.md` files with `## Section` headings into `docs/` and click **Rebuild index**.

## Stack
Python · Ollama (Llama 3.2, nomic-embed-text) · NumPy vector search · scikit-learn · Streamlit · pytest · optional Claude API

## Roadmap
- PDF / Word document upload
- Hybrid search (keywords + embeddings)
- Connect it to an MCP server so Claude Desktop can query the documents
