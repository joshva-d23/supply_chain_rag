# Meridian Supply Chain RAG Assistant

**Assignment 2 — Build a RAG System for Supply Chain Documents**

A clean Retrieval-Augmented Generation system that lets a buyer ask plain-English questions across two Meridian Components documents:

- `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (numbers, scorecards, incidents)
- `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (rules, clauses, classification, penalties)

Answers are grounded **only** in the uploaded documents. If the information is missing, the system refuses instead of inventing.

---

## Features

| Feature | Status |
|---------|--------|
| Upload one or more PDFs | ✅ |
| Index button with chunk count confirmation | ✅ |
| Persistent ChromaDB (survives restart) | ✅ |
| GPT-4o answers with file + page sources | ✅ |
| Honest refusal on out-of-scope questions | ✅ |
| Example questions (incl. trap question) | ✅ |
| Adjustable `top_k` | ✅ |
| Clear Index button | ✅ |
| Professional Streamlit UI | ✅ |

---

## Tech Stack (exact match to requirements)

| Component | Choice |
|-----------|--------|
| Language | Python 3.10+ |
| PDF reading | LangChain `PyPDFLoader` (pypdf) |
| Chunking | RecursiveCharacterTextSplitter · **size 1000 · overlap 150** |
| Embeddings | `text-embedding-3-small` |
| Vector DB | ChromaDB (persisted to `./chroma_db`) |
| LLM | GPT-4o · temperature 0.1 |
| Orchestration | LangChain |
| UI | Streamlit |
| Secrets | `.env` + `python-dotenv` |

**Why chunk size 1000 / overlap 150?**  
Financial and policy tables lose structure when split too aggressively. 1000 characters keeps most tables inside a single chunk while staying inside the mandated 800–1200 range. 150-character overlap preserves sentence continuity across boundaries.

---

## Project Structure

```
supplychain-rag/
├── app.py                 # Streamlit UI
├── ingest.py              # load → chunk → embed → store
├── rag.py                 # retrieve + prompt + GPT-4o
├── data/                  # the two provided PDFs
├── chroma_db/             # created automatically (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd supplychain-rag
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your OpenAI key

```bash
cp .env.example .env
# edit .env and put your real key
```

### 3. Run the app

```bash
streamlit run app.py
```

The two PDFs are already inside `data/`.  
Click **Index Documents** once, then start asking questions.

## Deploy on Render

1. Push this project folder to a GitHub, GitLab, or Bitbucket repository.
2. In Render, choose **New > Blueprint** and connect the repository.
3. Render will detect `render.yaml`. Enter `OPENAI_API_KEY` when prompted.
4. After deployment, open the generated `onrender.com` URL and click
   **Index Documents**.

The free Render filesystem is ephemeral, so the Chroma index can disappear after
a restart or redeploy. For persistence, upgrade the web service, attach a disk at
`/opt/render/project/src/chroma_db`, and set `CHROMA_DIR` to the same path.

---

## Test Questions & Sample Answers

Record the actual answers your running app produces. Below are the expected themes (verify against the PDFs).

1. **Highest spend supplier**  
   Shenzhen Rui Electronics · ₹21.9 crore · 79.5% on-time delivery

2. **Line stoppages**  
   7 events · 41 hours · mainly microcontroller shortages + two PCB rejections

3. **Approval for ₹1.4 crore PO**  
   Chief Operating Officer (above ₹1 crore and up to ₹5 crore)

4. **Four classification categories**  
   Critical / Strategic / Standard / Tail — Critical = single-source **or** spend > ₹10 crore **or** safety-related part

5. **Kaveri Metals 88.1% OTD + 1150 PPM**  
   Triggers clause 6.1 (OTD < 90%) and clause 6.3 (defect > 500 PPM). Buyer must issue written warning, move to weekly review, impose 100% inspection, and recover rework cost at ₹120/unit.

6. **Single-source microcontroller**  
   Policy 7.1 requires dual sourcing within 12 months. Company is already qualifying Anh Long Semiconductors (Vietnam) — target 30 Sep 2025.

7. **Safety stock for 46-day imported Critical part**  
   Calculated = 46 × 0.25 = 11.5 → floor for imported Critical = 30 days → **hold 30 days**.

8. **Trident 640 PPM**  
   Clause 6.3 applies → supplier bears rework at ₹120 per affected unit + 100% incoming inspection until three clean lots.

9. **Below B-band on OTD alone**  
   Any supplier with OTD < 75% cannot be band B. From the scorecard: Shenzhen Rui (79.5%) is still above 75%; none are below 75% on the published numbers. Escalation follows the matrix in section 10.

10. **Trap question**  
    “What is the annual salary of the Head of Procurement?”  
    → **The information is not available in the uploaded documents.**

---

## Screenshots

*(Add your own screenshots here after running the app)*

- Upload + Index success banner  
- Answer with source cards  
- Trap-question refusal  

---

## What worked well / What was harder

**Worked well**
- Persistent ChromaDB survived restarts without re-upload.
- Source cards with page numbers make verification easy.
- Example-question dropdown speeds up demos.

**Harder**
- Tables become linear text after extraction (expected). Larger chunk size helped.
- Cross-document questions needed `top_k ≥ 5`; with `top_k=3` the retriever sometimes returned only one document.
- Always print retrieved chunks when debugging — most “wrong” answers were retrieval issues, not model hallucinations.

---

## Optional FastAPI backend

Not implemented in this base version. The same `ingest.py` and `rag.py` functions can be wrapped behind FastAPI endpoints (`/ingest`, `/ask`, `/stats`) if you want the +15 bonus marks.

---

## License

For educational use only — Assignment 2 submission.
