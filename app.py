"""
Semantic Search AI System - FastAPI web app
===========================================

Turns your Colab notebook (NLTK preprocessing -> Word2Vec -> mean document
vectors -> cosine similarity) into a web application.

SETUP (run once in the VS Code terminal)
----------------------------------------
    pip install fastapi "uvicorn[standard]" python-multipart pandas numpy nltk gensim

RUN
---
    1. Put your dataset CSV in the same folder as this file.
       It needs the columns used in your notebook:
           document_id, title, content, keywords, category
       (only `title` and `content` are strictly required)
    2. python app.py
    3. Open http://127.0.0.1:8000

    Optional:
       python app.py --data my_dataset.csv --port 8080

If no CSV is found, the web page shows an upload box, so you can also just
drop the file there.
"""

import argparse
import io
import json
import random
import re
import string
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import nltk
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "word2vec_sematic.model"
MODEL_META_PATH = BASE_DIR / "word2vec_sematic.meta.json"

# --------------------------------------------------------------------------
# NLTK resources + text preprocessing (identical logic to the notebook)
# --------------------------------------------------------------------------
for _res in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_res, quiet=True)
    except Exception as exc:  # offline machines can still work if data exists
        print(f"[warn] could not download NLTK resource '{_res}': {exc}")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(w, pos="v"), pos="n") for w in tokens]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Search engine
# --------------------------------------------------------------------------
class SearchEngine:
    """Holds the dataset, Word2Vec model and document embeddings."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ready = False
        self.building = False
        self.message = "Starting up..."
        self.error: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[Word2Vec] = None
        self.doc_norm: Optional[np.ndarray] = None
        self.source = ""

    # ---- dataset helpers -------------------------------------------------
    @staticmethod
    def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in ("title", "content") if c not in df.columns]
        if missing:
            raise ValueError(
                "The CSV must contain the columns 'title' and 'content'. "
                f"Missing: {', '.join(missing)}. Found: {', '.join(df.columns)}"
            )
        if "keywords" not in df.columns:
            df["keywords"] = ""
        if "category" not in df.columns:
            df["category"] = "Uncategorized"
        if "document_id" not in df.columns:
            df["document_id"] = range(1, len(df) + 1)
        for col in ("title", "content", "keywords", "category"):
            df[col] = df[col].fillna("").astype(str)
        df["category"] = df["category"].replace("", "Uncategorized")
        return df.reset_index(drop=True)

    @staticmethod
    def find_csv() -> Optional[Path]:
        candidates = sorted(
            BASE_DIR.glob("*.csv"),
            key=lambda p: (p.name.startswith("preprocessed"), p.name.lower()),
        )
        for path in candidates:
            try:
                cols = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
            except Exception:
                continue
            if "title" in cols and "content" in cols:
                return path
        return None

    # ---- build index -----------------------------------------------------
    def build(self, df: pd.DataFrame, source: str, reuse_model: bool = True) -> None:
        with self.lock:
            self.building = True
            self.ready = False
            self.error = None
        try:
            df = self.validate_frame(df)

            self.message = "Cleaning and tokenising documents..."
            df["text"] = df["title"] + " " + df["content"] + " " + df["keywords"]
            if "clean_text" in df.columns and df["clean_text"].notna().all():
                df["clean_text"] = df["clean_text"].astype(str)
            else:
                df["clean_text"] = df["text"].apply(preprocess_text)

            signature = {"source": source, "documents": int(len(df))}
            model = None
            if reuse_model and MODEL_PATH.exists() and MODEL_META_PATH.exists():
                try:
                    if json.loads(MODEL_META_PATH.read_text()) == signature:
                        self.message = "Loading saved Word2Vec model..."
                        model = Word2Vec.load(str(MODEL_PATH))
                except Exception:
                    model = None

            if model is None:
                self.message = "Training Word2Vec model (skip-gram, 100 dimensions)..."
                corpus = df["clean_text"].apply(lambda x: x.split()).tolist()
                model = Word2Vec(
                    sentences=corpus,
                    vector_size=100,
                    window=5,
                    min_count=2,
                    workers=4,
                    sg=1,
                    epochs=20,
                    seed=42,
                )
                model.save(str(MODEL_PATH))
                MODEL_META_PATH.write_text(json.dumps(signature))

            self.message = "Creating document embeddings..."
            embeddings = np.array(
                [self._doc_vector(t, model) for t in df["clean_text"]], dtype=np.float32
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # zero vectors stay zero -> similarity 0

            with self.lock:
                self.df = df
                self.model = model
                self.doc_norm = embeddings / norms
                self.source = source
                self.ready = True
                self.message = "Ready"
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.message = "Could not build the search index"
            print(f"[error] {exc}")
        finally:
            with self.lock:
                self.building = False

    @staticmethod
    def _doc_vector(text: str, model: Word2Vec) -> np.ndarray:
        vectors = [model.wv[w] for w in text.split() if w in model.wv]
        if not vectors:
            return np.zeros(model.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def build_in_background(self, path: Optional[Path]) -> None:
        def job() -> None:
            target = path or self.find_csv()
            if target is None:
                self.message = "Waiting for a dataset"
                return
            try:
                self.message = f"Reading {target.name}..."
                df = pd.read_csv(target)
                self.build(df, target.name)
            except Exception as exc:
                self.error = str(exc)
                self.message = "Could not read the dataset"

        threading.Thread(target=job, daemon=True).start()

    # ---- search ----------------------------------------------------------
    def search(self, query: str, top_k: int, category: Optional[str]) -> dict:
        if not self.ready or self.df is None or self.model is None:
            raise HTTPException(status_code=503, detail="The search index is not ready yet.")

        wv = self.model.wv
        cleaned = preprocess_text(query)
        tokens = cleaned.split()
        known = [t for t in tokens if t in wv]
        unknown = [t for t in tokens if t not in wv]

        response = {
            "query": query,
            "matched_terms": known,
            "unknown_terms": unknown,
            "related_terms": [],
            "results": [],
            "total_candidates": 0,
        }
        if not known:
            return response

        q_vec = np.mean([wv[t] for t in known], axis=0).astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return response
        scores = self.doc_norm @ (q_vec / q_norm)

        if category and category != "All":
            mask = (self.df["category"] == category).to_numpy()
            candidate_idx = np.where(mask)[0]
        else:
            candidate_idx = np.arange(len(self.df))

        response["total_candidates"] = int(len(candidate_idx))
        if len(candidate_idx) == 0:
            return response

        order = candidate_idx[np.argsort(scores[candidate_idx])[::-1][:top_k]]

        related, seen = [], set(known)
        for term in known:
            for word, _ in wv.most_similar(term, topn=5):
                if word not in seen:
                    seen.add(word)
                    related.append(word)
        response["related_terms"] = related[:8]

        for rank, idx in enumerate(order, start=1):
            row = self.df.iloc[int(idx)]
            snippet = re.sub(r"\s+", " ", row["content"]).strip()
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "..."
            response["results"].append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "document_id": _plain(row["document_id"]),
                    "title": row["title"],
                    "category": row["category"],
                    "keywords": row["keywords"],
                    "snippet": snippet,
                    "score": round(float(scores[idx]), 4),
                }
            )
        return response

    def status(self) -> dict:
        info = {
            "ready": self.ready,
            "building": self.building,
            "message": self.message,
            "error": self.error,
            "source": self.source,
            "documents": 0,
            "vocabulary": 0,
            "dimensions": 0,
            "categories": [],
            "examples": [],
        }
        if self.ready and self.df is not None and self.model is not None:
            info["documents"] = int(len(self.df))
            info["vocabulary"] = int(len(self.model.wv))
            info["dimensions"] = int(self.model.vector_size)
            info["categories"] = sorted(self.df["category"].unique().tolist())
            titles = [t for t in self.df["title"].tolist() if 12 <= len(t) <= 70]
            info["examples"] = random.sample(titles, k=min(4, len(titles)))
        return info


def _plain(value):
    """Convert numpy scalars to plain Python values for JSON."""
    return value.item() if hasattr(value, "item") else value


engine = SearchEngine()

# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(title="Semantic Search AI System")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category: Optional[str] = None


@app.on_event("startup")
def on_startup() -> None:
    engine.build_in_background(CLI_DATA_PATH)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


@app.get("/api/status")
def api_status() -> dict:
    return engine.status()


@app.post("/api/search")
def api_search(req: SearchRequest) -> dict:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a search query.")
    top_k = max(1, min(int(req.top_k), 50))
    return engine.search(query, top_k, req.category)


@app.get("/api/document/{index}")
def api_document(index: int) -> dict:
    if not engine.ready or engine.df is None:
        raise HTTPException(status_code=503, detail="The search index is not ready yet.")
    if index < 0 or index >= len(engine.df):
        raise HTTPException(status_code=404, detail="Document not found.")
    row = engine.df.iloc[index]
    return {
        "document_id": _plain(row["document_id"]),
        "title": row["title"],
        "category": row["category"],
        "keywords": row["keywords"],
        "content": row["content"],
    }


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    if engine.building:
        raise HTTPException(status_code=409, detail="An index is already being built. Please wait.")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
        SearchEngine.validate_frame(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    def job() -> None:
        engine.build(df, file.filename, reuse_model=False)

    engine.message = "Reading your dataset..."
    engine.building = True
    engine.ready = False
    threading.Thread(target=job, daemon=True).start()
    return {"accepted": True, "documents": int(len(df))}


# --------------------------------------------------------------------------
# Front end (single page, served from memory)
# --------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #0b181d;
    --panel: #11242b;
    --panel-hi: #163039;
    --line: #21404a;
    --text: #e8f1ee;
    --muted: #8fa9a8;
    --amber: #f2b84b;
    --mint: #79d2b8;
    --danger: #ff8a7a;
    --serif: "Fraunces", Georgia, "Times New Roman", serif;
    --sans: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    background: var(--ink);
    color: var(--text);
    font-family: var(--sans);
    font-size: 16px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  a, button, input, select { font: inherit; color: inherit; }
  :focus-visible { outline: 2px solid var(--amber); outline-offset: 3px; border-radius: 6px; }

  .wrap { max-width: 920px; margin: 0 auto; padding: 0 24px; }

  /* ---------- top bar ---------- */
  header.bar {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 22px 0; border-bottom: 1px solid var(--line);
  }
  .brand { font-family: var(--serif); font-size: 1.25rem; font-weight: 600; letter-spacing: .01em; }
  .stats { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: .88rem; flex-wrap: wrap; justify-content: flex-end; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted); flex: none; }
  .dot.ok { background: var(--mint); }
  .dot.busy { background: var(--amber); animation: pulse 1.2s ease-in-out infinite; }
  .dot.err { background: var(--danger); }
  @keyframes pulse { 50% { opacity: .35; } }

  /* ---------- hero ---------- */
  .hero { padding: 72px 0 28px; }
  h1 {
    font-family: var(--serif); font-weight: 400; font-size: clamp(2.3rem, 6.4vw, 4rem);
    line-height: 1.06; letter-spacing: -.02em; margin: 0 0 16px; max-width: 14ch;
  }
  .lede { color: var(--muted); max-width: 52ch; margin: 0 0 34px; font-size: 1.05rem; }

  .searchbox {
    display: flex; align-items: center; gap: 8px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
    padding: 8px 8px 8px 20px; transition: border-color .2s, box-shadow .2s;
  }
  .searchbox:focus-within { border-color: var(--amber); box-shadow: 0 0 0 4px rgba(242,184,75,.12); }
  .searchbox input {
    flex: 1; min-width: 0; background: none; border: 0; outline: 0;
    font-size: 1.15rem; padding: 12px 0;
  }
  .searchbox input::placeholder { color: #5f7c7d; }
  .go {
    background: var(--amber); color: #2a1d00; border: 0; border-radius: 11px;
    padding: 13px 26px; font-weight: 600; cursor: pointer; transition: transform .12s, filter .2s;
  }
  .go:hover { filter: brightness(1.07); }
  .go:active { transform: scale(.97); }
  .go:disabled { opacity: .45; cursor: not-allowed; }

  .controls { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; margin-top: 16px; color: var(--muted); font-size: .92rem; }
  .controls label { display: flex; align-items: center; gap: 8px; }
  .controls select {
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 6px 10px; color: var(--text); max-width: 220px;
  }
  .hint kbd { background: var(--panel-hi); border: 1px solid var(--line); border-radius: 5px; padding: 0 6px; font-family: inherit; font-size: .85em; }

  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    background: transparent; border: 1px solid var(--line); border-radius: 999px;
    padding: 5px 13px; font-size: .88rem; color: var(--muted); cursor: pointer;
    transition: border-color .15s, color .15s, background .15s;
  }
  .chip:hover { border-color: var(--mint); color: var(--text); background: rgba(121,210,184,.07); }
  .try { display: flex; gap: 12px; align-items: baseline; margin-top: 22px; flex-wrap: wrap; }
  .try span { color: var(--muted); font-size: .9rem; }

  /* ---------- states ---------- */
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 28px; margin: 28px 0; }
  .panel h2 { font-family: var(--serif); font-weight: 400; font-size: 1.5rem; margin: 0 0 6px; }
  .panel p { margin: 0 0 14px; color: var(--muted); }
  .drop {
    display: block; border: 1.5px dashed var(--line); border-radius: 12px; padding: 34px 20px;
    text-align: center; cursor: pointer; color: var(--muted); transition: border-color .2s, background .2s;
  }
  .drop:hover, .drop.over { border-color: var(--amber); background: rgba(242,184,75,.05); color: var(--text); }
  .drop input { display: none; }
  .err-text { color: var(--danger); margin-top: 12px; font-size: .92rem; min-height: 1.2em; }
  .bar-track { height: 4px; background: var(--line); border-radius: 4px; overflow: hidden; margin-top: 18px; }
  .bar-fill { height: 100%; width: 35%; background: var(--amber); border-radius: 4px; animation: slide 1.4s ease-in-out infinite; }
  @keyframes slide { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }

  /* ---------- understanding strip ---------- */
  .read { margin: 34px 0 6px; padding-bottom: 18px; border-bottom: 1px solid var(--line); }
  .read h2 { font-family: var(--serif); font-weight: 400; font-size: 1.5rem; margin: 0 0 12px; }
  .read h2 em { font-style: normal; color: var(--amber); }
  .readrow { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; font-size: .9rem; color: var(--muted); }
  .term { border-radius: 6px; padding: 2px 9px; background: var(--panel-hi); color: var(--text); font-size: .86rem; }
  .term.skip { text-decoration: line-through; color: var(--muted); background: transparent; border: 1px solid var(--line); }

  /* ---------- results ---------- */
  .results { list-style: none; margin: 0; padding: 0; }
  .row {
    display: grid; grid-template-columns: 44px 1fr 168px; gap: 20px;
    padding: 26px 0; border-bottom: 1px solid var(--line); cursor: pointer;
  }
  .row:hover .title { color: var(--amber); }
  .rank { font-family: var(--serif); font-size: 1.7rem; color: #4d6c70; line-height: 1.1; }
  .title { font-family: var(--serif); font-weight: 600; font-size: 1.3rem; line-height: 1.25; margin: 0 0 6px; transition: color .15s; }
  .cat { display: inline-block; font-size: .8rem; color: var(--mint); margin-bottom: 8px; }
  .snippet { color: #b5c9c6; margin: 0 0 10px; max-width: 62ch; font-size: .96rem; }
  .kw { color: var(--muted); font-size: .84rem; }
  .meter { padding-top: 6px; }
  .ticks { display: grid; grid-template-columns: repeat(20, 1fr); gap: 2px; height: 22px; }
  .ticks i { background: var(--line); border-radius: 1px; }
  .ticks i.on { background: var(--amber); }
  .score { margin-top: 8px; font-size: .84rem; color: var(--muted); display: flex; justify-content: space-between; }
  .score b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }

  .empty { padding: 44px 0; color: var(--muted); }
  .empty strong { display: block; color: var(--text); font-family: var(--serif); font-weight: 400; font-size: 1.4rem; margin-bottom: 6px; }

  /* ---------- document dialog ---------- */
  dialog {
    width: min(720px, calc(100% - 32px)); max-height: 86vh; padding: 0; border: 1px solid var(--line);
    border-radius: 18px; background: var(--panel); color: var(--text);
  }
  dialog::backdrop { background: rgba(4,10,12,.72); }
  dialog[open] { animation: rise .22s ease-out; }
  @keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
  .dlg-head { padding: 26px 28px 8px; }
  .dlg-head h3 { font-family: var(--serif); font-weight: 600; font-size: 1.6rem; line-height: 1.2; margin: 6px 0 0; }
  .dlg-body { padding: 8px 28px 8px; overflow-y: auto; max-height: 52vh; color: #c4d6d3; white-space: pre-wrap; }
  .dlg-foot { display: flex; gap: 10px; justify-content: space-between; align-items: center; padding: 18px 28px 24px; flex-wrap: wrap; }
  .ghost { background: transparent; border: 1px solid var(--line); border-radius: 10px; padding: 9px 16px; cursor: pointer; transition: border-color .15s; }
  .ghost:hover { border-color: var(--mint); }
  .solid { background: var(--amber); color: #2a1d00; border: 0; border-radius: 10px; padding: 10px 18px; font-weight: 600; cursor: pointer; }

  footer { padding: 40px 0 56px; color: var(--muted); font-size: .86rem; display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  footer button { background: none; border: 0; color: var(--muted); text-decoration: underline; cursor: pointer; padding: 0; }
  footer button:hover { color: var(--text); }

  [hidden] { display: none !important; }

  @media (max-width: 680px) {
    .hero { padding-top: 44px; }
    .row { grid-template-columns: 1fr; gap: 10px; }
    .rank { font-size: 1.2rem; }
    .meter { max-width: 220px; }
    .go { padding: 13px 18px; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header class="bar">
    <div class="brand">Semantic Search</div>
    <div class="stats" id="stats" aria-live="polite">
      <span class="dot busy" id="dot"></span><span id="statText">Starting up...</span>
    </div>
  </header>

  <section class="hero">
    <h1>Search by meaning, not keywords.</h1>
    <p class="lede">Describe what you are looking for in your own words. Documents are ranked by how close their meaning is to your query, using a Word2Vec model trained on your dataset.</p>

    <form class="searchbox" id="form" role="search">
      <input id="q" type="text" autocomplete="off" spellcheck="false" aria-label="Search query"
             placeholder="e.g. how do vaccines protect against disease" disabled>
      <button class="go" id="goBtn" type="submit" disabled>Search</button>
    </form>

    <div class="controls">
      <label>Results
        <select id="topk" aria-label="Number of results">
          <option>5</option><option>10</option><option>20</option><option>50</option>
        </select>
      </label>
      <label>Category
        <select id="cat" aria-label="Filter by category"><option value="All">All categories</option></select>
      </label>
      <span class="hint">Press <kbd>/</kbd> to jump to the search box</span>
    </div>

    <div class="try" id="tryRow" hidden>
      <span>Try a title from your data:</span>
      <div class="chips" id="examples"></div>
    </div>
  </section>

  <!-- building / upload states -->
  <section class="panel" id="buildPanel" hidden>
    <h2 id="buildTitle">Building the search index</h2>
    <p id="buildMsg">This takes a short while the first time. The trained model is saved, so the next start is quicker.</p>
    <div class="bar-track"><div class="bar-fill"></div></div>
  </section>

  <section class="panel" id="uploadPanel" hidden>
    <h2>Add your dataset</h2>
    <p>No CSV was found next to <code>app.py</code>. Upload the file you used in your notebook. It needs a <code>title</code> and a <code>content</code> column; <code>keywords</code>, <code>category</code> and <code>document_id</code> are used when present.</p>
    <label class="drop" id="drop">
      <input type="file" id="file" accept=".csv">
      <span id="dropText">Drop a CSV here, or click to choose a file</span>
    </label>
    <div class="err-text" id="uploadErr" role="alert"></div>
  </section>

  <!-- results -->
  <main id="out" aria-live="polite"></main>

  <footer>
    <span id="sourceText"></span>
    <button type="button" id="replaceBtn" hidden>Use a different dataset</button>
  </footer>
</div>

<dialog id="dlg" aria-labelledby="dlgTitle">
  <div class="dlg-head">
    <span class="cat" id="dlgMeta"></span>
    <h3 id="dlgTitle"></h3>
  </div>
  <div class="dlg-body" id="dlgBody"></div>
  <div class="dlg-foot">
    <span class="kw" id="dlgKw"></span>
    <div style="display:flex; gap:10px">
      <button class="ghost" type="button" id="dlgClose">Close</button>
      <button class="solid" type="button" id="dlgSimilar">Find similar documents</button>
    </div>
  </div>
</dialog>

<script>
const $ = (id) => document.getElementById(id);
const els = {
  q: $('q'), go: $('goBtn'), form: $('form'), topk: $('topk'), cat: $('cat'),
  out: $('out'), dot: $('dot'), statText: $('statText'),
  buildPanel: $('buildPanel'), buildMsg: $('buildMsg'), buildTitle: $('buildTitle'),
  uploadPanel: $('uploadPanel'), uploadErr: $('uploadErr'),
  file: $('file'), drop: $('drop'), dropText: $('dropText'),
  tryRow: $('tryRow'), examples: $('examples'),
  sourceText: $('sourceText'), replaceBtn: $('replaceBtn'),
  dlg: $('dlg'), dlgTitle: $('dlgTitle'), dlgMeta: $('dlgMeta'), dlgBody: $('dlgBody'),
  dlgKw: $('dlgKw'), dlgClose: $('dlgClose'), dlgSimilar: $('dlgSimilar'),
};

let wasReady = false;
let pollTimer = null;
let currentDoc = null;

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ---------------- status polling ---------------- */
async function refreshStatus() {
  let s;
  try {
    s = await (await fetch('/api/status')).json();
  } catch (e) {
    setDot('err', 'Cannot reach the server');
    return schedulePoll(3000);
  }

  if (s.ready) {
    setDot('ok', `${s.documents.toLocaleString()} documents · ${s.vocabulary.toLocaleString()} words · ${s.dimensions}-d vectors`);
    els.buildPanel.hidden = true;
    els.uploadPanel.hidden = true;
    els.q.disabled = false; els.go.disabled = false;
    els.replaceBtn.hidden = false;
    els.sourceText.textContent = `Dataset: ${s.source}`;
    if (!wasReady) {
      fillCategories(s.categories);
      fillExamples(s.examples);
      els.q.focus();
    }
    wasReady = true;
    return;
  }

  wasReady = false;
  els.q.disabled = true; els.go.disabled = true;

  if (s.error) {
    setDot('err', 'Something went wrong');
    els.buildPanel.hidden = true;
    els.uploadPanel.hidden = false;
    els.uploadErr.textContent = s.error;
    return schedulePoll(4000);
  }
  if (s.building) {
    setDot('busy', s.message);
    els.uploadPanel.hidden = true;
    els.buildPanel.hidden = false;
    els.buildMsg.textContent = s.message;
    return schedulePoll(1200);
  }
  // idle and no dataset yet
  setDot('busy', 'Waiting for a dataset');
  els.buildPanel.hidden = true;
  els.uploadPanel.hidden = false;
  schedulePoll(2500);
}
function setDot(kind, text) {
  els.dot.className = 'dot ' + kind;
  els.statText.textContent = text;
}
function schedulePoll(ms) { clearTimeout(pollTimer); pollTimer = setTimeout(refreshStatus, ms); }

function fillCategories(cats) {
  els.cat.innerHTML = '<option value="All">All categories</option>' +
    cats.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
}
function fillExamples(list) {
  if (!list.length) { els.tryRow.hidden = true; return; }
  els.tryRow.hidden = false;
  els.examples.innerHTML = list.map((t) => `<button type="button" class="chip">${esc(t)}</button>`).join('');
  els.examples.querySelectorAll('.chip').forEach((b) =>
    b.addEventListener('click', () => runSearch(b.textContent)));
}

/* ---------------- search ---------------- */
async function runSearch(text) {
  const query = (text ?? els.q.value).trim();
  if (!query) { els.q.focus(); return; }
  els.q.value = query;
  els.go.disabled = true; els.go.textContent = 'Searching...';
  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: +els.topk.value, category: els.cat.value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Search failed');
    render(data);
    els.out.scrollIntoView({ block: 'start' });
  } catch (e) {
    els.out.innerHTML = `<div class="empty"><strong>The search did not run</strong>${esc(e.message)}</div>`;
  } finally {
    els.go.disabled = false; els.go.textContent = 'Search';
  }
}

function termChips(list, cls = '') {
  return list.map((t) => `<span class="term ${cls}">${esc(t)}</span>`).join('');
}

function meter(score) {
  const on = Math.max(0, Math.min(20, Math.round(score * 20)));
  let ticks = '';
  for (let i = 0; i < 20; i++) ticks += `<i class="${i < on ? 'on' : ''}"></i>`;
  return `<div class="ticks" role="img" aria-label="Similarity ${score.toFixed(2)} out of 1">${ticks}</div>
          <div class="score"><span>similarity</span><b>${score.toFixed(4)}</b></div>`;
}

function render(d) {
  let html = '';
  const n = d.results.length;
  html += `<div class="read"><h2>${n} ${n === 1 ? 'result' : 'results'} for <em>${esc(d.query)}</em></h2>`;
  if (d.matched_terms.length || d.unknown_terms.length) {
    html += `<div class="readrow"><span>Understood as</span>${termChips(d.matched_terms)}`;
    if (d.unknown_terms.length) html += `<span>Not in vocabulary</span>${termChips(d.unknown_terms, 'skip')}`;
    html += `</div>`;
  }
  if (d.related_terms.length) {
    html += `<div class="readrow"><span>Related words</span><div class="chips" id="related">` +
      d.related_terms.map((t) => `<button type="button" class="chip">${esc(t)}</button>`).join('') +
      `</div></div>`;
  }
  html += `</div>`;

  if (!n) {
    const why = d.matched_terms.length
      ? 'No documents are in the selected category. Choose "All categories" and try again.'
      : 'None of the words in your query appear in the model\u2019s vocabulary. Try different or more common words from your documents.';
    html += `<div class="empty"><strong>Nothing to show</strong>${why}</div>`;
    els.out.innerHTML = html;
    return;
  }

  html += '<ol class="results">' + d.results.map((r) => `
    <li class="row" tabindex="0" data-index="${r.index}">
      <div class="rank">${r.rank}</div>
      <div>
        <h3 class="title">${esc(r.title)}</h3>
        <span class="cat">${esc(r.category)}</span>
        <p class="snippet">${esc(r.snippet)}</p>
        ${r.keywords ? `<div class="kw">${esc(r.keywords)}</div>` : ''}
      </div>
      <div class="meter">${meter(r.score)}</div>
    </li>`).join('') + '</ol>';
  els.out.innerHTML = html;

  els.out.querySelectorAll('.row').forEach((row) => {
    const open = () => openDoc(+row.dataset.index);
    row.addEventListener('click', open);
    row.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(); });
  });
  const rel = $('related');
  if (rel) rel.querySelectorAll('.chip').forEach((b) =>
    b.addEventListener('click', () => runSearch(`${d.query} ${b.textContent}`)));
}

/* ---------------- document dialog ---------------- */
async function openDoc(index) {
  try {
    const res = await fetch('/api/document/' + index);
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || 'Could not load the document');
    currentDoc = d;
    els.dlgMeta.textContent = `${d.category} · ID ${d.document_id}`;
    els.dlgTitle.textContent = d.title;
    els.dlgBody.textContent = d.content;
    els.dlgKw.textContent = d.keywords ? 'Keywords: ' + d.keywords : '';
    els.dlg.showModal();
  } catch (e) { alert(e.message); }
}
els.dlgClose.addEventListener('click', () => els.dlg.close());
els.dlg.addEventListener('click', (e) => { if (e.target === els.dlg) els.dlg.close(); });
els.dlgSimilar.addEventListener('click', () => {
  if (!currentDoc) return;
  els.dlg.close();
  runSearch(currentDoc.title);
});

/* ---------------- upload ---------------- */
async function upload(file) {
  if (!file) return;
  els.uploadErr.textContent = '';
  els.dropText.textContent = `Uploading ${file.name}...`;
  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    els.out.innerHTML = '';
    els.dropText.textContent = 'Drop a CSV here, or click to choose a file';
    wasReady = false;
    els.uploadPanel.hidden = true;
    els.buildPanel.hidden = false;
    els.buildTitle.textContent = 'Building the search index';
    refreshStatus();
  } catch (e) {
    els.uploadErr.textContent = e.message;
    els.dropText.textContent = 'Drop a CSV here, or click to choose a file';
  }
}
els.file.addEventListener('change', () => upload(els.file.files[0]));
['dragenter', 'dragover'].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.remove('over'); }));
els.drop.addEventListener('drop', (e) => upload(e.dataTransfer.files[0]));
els.replaceBtn.addEventListener('click', () => {
  els.uploadPanel.hidden = !els.uploadPanel.hidden;
  if (!els.uploadPanel.hidden) els.uploadPanel.scrollIntoView({ behavior: 'smooth' });
});

/* ---------------- wiring ---------------- */
els.form.addEventListener('submit', (e) => { e.preventDefault(); runSearch(); });
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== els.q && !els.dlg.open && !els.q.disabled) {
    e.preventDefault(); els.q.focus();
  }
});
refreshStatus();
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
CLI_DATA_PATH: Optional[Path] = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search AI System")
    parser.add_argument("--data", type=str, default=None, help="Path to the dataset CSV")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.data:
        CLI_DATA_PATH = Path(args.data).expanduser().resolve()
        if not CLI_DATA_PATH.exists():
            raise SystemExit(f"Dataset not found: {CLI_DATA_PATH}")

    print(f"\n  Semantic Search is starting at http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)
