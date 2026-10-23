"""Live demo front end.

Runs the RAG pipeline IN-PROCESS (Streamlit Community Cloud can't host a separate API server).
Set RAG_API_URL to use the FastAPI service instead. Without an LLM key it falls back to the
extractive generator, so the app still works (verbatim passages, no synthesis).
"""
import datetime, os, pathlib, subprocess, sys, threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)                     # data/raw and artifacts/ resolve from the repo root

import streamlit as st

ABOUT = """**scikit-learn docs assistant**: ask a question and get an answer built only from 21 pages of the scikit-learn 1.5.2 user guide, with the passages it used shown as sources. If the answer isn't in the docs, it says so instead of guessing.

Under the hood: BGE embeddings and BM25 hybrid retrieval, a cross-encoder reranker, and an LLM that must cite its sources.

The point of the project is the **evaluation harness**: Recall@k and MRR ablations (chunk size, reranking, hybrid search), citation checks and refusal testing on questions the documentation can't answer.

Code and results: https://github.com/Tejashree-Likhar/Domain-RAG-assistant-with-a-proper-evaluation-harness"""

st.set_page_config(page_title="scikit-learn docs assistant", page_icon="📚", layout="wide",
                   menu_items={"About": ABOUT, "Get help": None, "Report a bug": None})

SECRET_KEYS = ["RAG_GENERATOR", "RAG_OPENAI_BASE_URL", "RAG_OPENAI_API_KEY", "RAG_MAX_RETRIES", "RAG_MAX_TOKENS",
               "RAG_MIN_INTERVAL_S", "RAG_EMBEDDER", "RAG_RERANKER", "DAILY_CAP", "SESSION_CAP"]


def load_secrets():
    """Streamlit secrets -> environment variables (the library reads env vars)."""
    try:
        for k in SECRET_KEYS:
            if k in st.secrets and k not in os.environ:
                os.environ[k] = str(st.secrets[k])
    except Exception:              # no secrets file: fine, env vars or defaults apply
        pass


load_secrets()

EXAMPLES = {
    "Quick facts": [
        "Which two parameters define what counts as 'dense' in DBSCAN?",
        "Between which values does the silhouette coefficient range?",
        "Can the R² score be negative?",
        "How does SVC deal with multi-class classification?",
        "Which parameter controls minimal cost-complexity pruning?",
        "What does TruncatedSVD compute?",
        "How does KNNImputer fill in missing values?",
        "Does scikit-learn's MLP run on GPUs?",
    ],
    "In your own words": [
        "How do I stop a decision tree from memorising the training data?",
        "What should I do about missing values in my features?",
        "How can I tune hyperparameters without trying every combination?",
        "My classifier's probabilities aren't trustworthy. How can I fix that?",
        "How do I keep test data from leaking into preprocessing?",
        "Which metric should I use when my classes are imbalanced?",
    ],
    "Should be refused (not in the docs)": [
        "How do Gaussian processes work in scikit-learn?",
        "How do I plot partial dependence?",
        "How do I fine-tune a BERT model using scikit-learn?",
        "How do I deploy a scikit-learn model on AWS SageMaker?",
        "What is the capital of Scotland?",
    ],
}


REFUSAL_TEXT = {
    "model_not_found": "The documentation doesn't appear to cover this, so the assistant won't guess.",
    "retrieval_gate": "The closest passages scored too low to answer from, so the assistant won't guess.",
    "no_overlap": "Nothing in the retrieved passages matches the question, so the assistant won't guess.",
}


def ensure_corpus():
    if not list((ROOT / "data" / "raw").glob("*.rst")):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_corpus.py")], check=True)


@st.cache_resource(show_spinner="Loading models and building the index (first load takes a minute or two)...")
def get_pipeline():
    from ragkit.pipeline import Config, build_pipeline
    ensure_corpus()
    cfg = Config(embedder=os.getenv("RAG_EMBEDDER", "BAAI/bge-small-en-v1.5"),
                 reranker=os.getenv("RAG_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                 mode="hybrid", chunk_tokens=400, k=5,
                 generator=os.getenv("RAG_GENERATOR", "extractive"))
    return build_pipeline(cfg), cfg


@st.cache_resource
def usage_counter():
    return {"day": None, "n": 0, "lock": threading.Lock()}


def take_daily_budget(cap: int) -> bool:
    u, today = usage_counter(), datetime.date.today()
    with u["lock"]:
        if u["day"] != today:
            u["day"], u["n"] = today, 0
        if u["n"] >= cap:
            return False
        u["n"] += 1
        return True


def ask(question: str, k: int):
    api = os.getenv("RAG_API_URL")
    if api:
        import requests
        r = requests.post(f"{api}/ask", json={"question": question, "k": k}, timeout=120)
        r.raise_for_status()
        return r.json()
    pipe, _ = get_pipeline()
    return pipe.answer(question, k)


# ---------------- UI ----------------
st.title("📚 scikit-learn docs assistant")
st.caption("Answers come only from 21 pages of the scikit-learn 1.5.2 user guide (BSD-3). "
           "If the answer isn't there, it should say so.")

pipe_cfg = None
if not os.getenv("RAG_API_URL"):
    _, pipe_cfg = get_pipeline()
uses_llm = bool(os.getenv("RAG_API_URL")) or (pipe_cfg and pipe_cfg.generator != "extractive")
if not uses_llm:
    st.info("No LLM key is configured, so this runs in **extractive mode**: it returns the best-matching "
            "sentences from the docs verbatim instead of writing an answer.")

DAILY_CAP, SESSION_CAP = int(os.getenv("DAILY_CAP", "150")), int(os.getenv("SESSION_CAP", "10"))
st.session_state.setdefault("asked", 0)


def pick(text):
    st.session_state["q"] = text
    st.session_state["go"] = True


with st.sidebar:
    st.header("Try a question")
    for group, qs in EXAMPLES.items():
        with st.expander(group, expanded=(group == "Quick facts")):
            for i, q in enumerate(qs):
                st.button(q, key=f"ex_{group}_{i}", on_click=pick, args=(q,), use_container_width=True)
    k = st.slider("Passages retrieved (k)", 1, 10, 5)
    if uses_llm:
        st.caption(f"Demo limits: {SESSION_CAP} questions per visit, {DAILY_CAP} per day overall.")

question = st.text_input("Ask a question about scikit-learn", key="q",
                         placeholder="Which two parameters define density in DBSCAN?")
go = st.button("Ask", type="primary") or st.session_state.pop("go", False)

if go and question.strip():
    if uses_llm and st.session_state["asked"] >= SESSION_CAP:
        st.warning("You've reached the per-visit question limit for this demo. Reload the page to continue.")
        st.stop()
    if uses_llm and not os.getenv("RAG_API_URL") and not take_daily_budget(DAILY_CAP):
        st.warning("Today's demo quota is used up. Please try again tomorrow, or run the project locally (see the GitHub README).")
        st.stop()
    st.session_state["asked"] += 1
    with st.spinner("Retrieving, reranking and answering..."):
        try:
            r = ask(question, k)
        except Exception as e:
            st.error("The answer step failed. If this is a rate-limit or quota problem, try again in a minute.")
            with st.expander("Technical details"):
                st.code(str(e)[:800])
            st.stop()
    refused = r["refused"]
    (st.warning if refused else st.success)(r["answer"])
    if refused:
        st.caption(REFUSAL_TEXT.get(r["refusal_reason"], "The assistant won't answer from outside the documentation."))
    lat = r["latency_ms"]
    st.caption(f"retrieve {lat['retrieve_ms']:.0f} ms · rerank {lat['rerank_ms']:.0f} ms · "
               f"generate {lat['generate_ms']:.0f} ms · total {lat['total_ms']:.0f} ms")
    if refused:
        with st.expander("Closest passages found (not used for an answer)"):
            for i, c in enumerate(r["citations"], 1):
                st.markdown(f"**[{i}] {c['section']}** · score {c['score']:.2f}")
                st.text(c["text"])
    else:
        st.subheader("Sources")
        st.caption(f"The {len(r['citations'])} passages retrieved. ✅ marks the ones the answer cites. "
                   "Check that they really support the claim.")
        for i, c in enumerate(r["citations"], 1):
            with st.expander(f"[{i}] {c['section']}  ·  score {c['score']:.2f}" + ("  ✅ cited" if c["cited"] else "")):
                st.text(c["text"])
