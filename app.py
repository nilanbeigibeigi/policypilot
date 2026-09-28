"""PolicyPilot — Streamlit chat UI.   Run:  streamlit run app.py"""
import os

import streamlit as st

import rag

st.set_page_config(page_title="PolicyPilot", page_icon="🧭")

# ---------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    llm = st.selectbox("Answer model", ["ollama", "claude", "none"],
                       help="ollama = free & local · claude = needs ANTHROPIC_API_KEY · none = retrieval only")
    embedder = st.selectbox("Search method", ["ollama", "tfidf"],
                            help="ollama = semantic embeddings (nomic-embed-text) · tfidf = keyword, offline")
    k = st.slider("Sources per answer (top-k)", 1, 6, 3)
    if st.button("🔄 Rebuild index"):
        st.cache_resource.clear()
    st.divider()
    st.caption("Documents in the library:")
    for c in sorted({c["title"] for c in rag.load_chunks()}):
        st.caption(f"📄 {c}")


@st.cache_resource(show_spinner="Indexing documents…")
def get_index(name: str):
    return rag.Index.build(name)


# ---------------- main
st.title("🧭 PolicyPilot")
st.caption("Ask anything about Saffron Market's policies. Answers come only from company documents, with sources.")

try:
    index = get_index(embedder)
except Exception as e:
    st.error(f"Couldn't build the index with **{embedder}**: {e}\n\n"
             "Is Ollama running? Run `ollama pull nomic-embed-text`, or switch Search method to **tfidf**.")
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []

examples = ["How many sick days do I get?", "Can a customer return opened saffron?",
            "What temperature should the fridge be?", "What is the WiFi password?"]
cols = st.columns(len(examples))
clicked = next((q for col, q in zip(cols, examples) if col.button(q, use_container_width=True)), None)


def show_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['title']} › {s['section']}** · similarity {s['score']}")
            st.caption(s["text"])


for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask a policy question…") or clicked
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents…"):
                r = rag.answer(index, question, k=k, llm=llm)
            st.markdown(r["answer"])
            show_sources(r["sources"])
            st.session_state.chat.append({"role": "assistant", "content": r["answer"], "sources": r["sources"]})
        except Exception as e:
            st.error(f"The **{llm}** model failed: {e}\n\n"
                     "For Ollama: is it running and did you `ollama pull llama3.2`? "
                     "For Claude: is ANTHROPIC_API_KEY set?")
