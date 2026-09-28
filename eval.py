"""Evaluate PolicyPilot on a fixed question set.

Measures:
  retrieval hit rate  - is the correct section in the top-k results?
  answer accuracy     - does the answer contain the expected fact?
  refusal accuracy    - for unanswerable questions, does it say "I don't know"?

Usage:
  python eval.py                  # Ollama embeddings + Ollama LLM
  LLM=claude python eval.py       # Claude as the LLM
  LLM=none python eval.py         # retrieval only (fast, no LLM)
"""
import json
import os
import time

import rag

K = int(os.environ.get("TOP_K", 3))
LLM = os.environ.get("LLM", "ollama")


def run(index=None, llm=LLM, k=K, verbose=True):
    questions = json.load(open(os.path.join(rag.HERE, "eval_questions.json")))
    index = index or rag.Index.build()
    rows, t0 = [], time.time()
    for item in questions:
        answerable = item["section"] is not None
        sources = index.search(item["q"], k)
        hit = any(s["section"] == item["section"] for s in sources) if answerable else None
        ans = rag.generate(item["q"], sources, llm) if llm != "none" else ""
        refused = rag.NO_ANSWER.lower().rstrip(".") in ans.lower()
        if llm == "none":
            correct = None
        elif answerable:
            correct = (not refused) and any(kw.lower() in ans.lower() for kw in item["keywords"])
        else:
            correct = refused
        rows.append({"q": item["q"], "answerable": answerable, "hit": hit, "correct": correct, "answer": ans})

    def pct(vals):
        vals = [v for v in vals if v is not None]
        return round(100 * sum(vals) / len(vals), 1) if vals else None

    summary = {
        "embedder": index.embedder.name, "llm": llm, "top_k": k, "questions": len(rows),
        "retrieval_hit_rate": pct([r["hit"] for r in rows]),
        "answer_accuracy": pct([r["correct"] for r in rows if r["answerable"]]),
        "refusal_accuracy": pct([r["correct"] for r in rows if not r["answerable"]]),
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        _write_report(summary, rows)
    return summary, rows


def _write_report(summary, rows):
    mark = {True: "✅", False: "❌", None: "—"}
    lines = ["# PolicyPilot evaluation", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v}{'%' if 'rate' in k or 'accuracy' in k else ''} |" for k, v in summary.items()]
    lines += ["", "| Question | Retrieval | Answer | Model answer |", "|---|---|---|---|"]
    for r in rows:
        a = r["answer"].replace("\n", " ").replace("|", "/")[:160]
        lines.append(f"| {r['q']} | {mark[r['hit']]} | {mark[r['correct']]} | {a} |")
    os.makedirs(os.path.join(rag.HERE, "reports"), exist_ok=True)
    path = os.path.join(rag.HERE, "reports", f"eval_{summary['embedder']}_{summary['llm']}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report: {path}")
    for r in rows:
        if r["hit"] is False:
            print("  retrieval miss:", r["q"])


if __name__ == "__main__":
    run()
