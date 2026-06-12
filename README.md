# RadMatcher

**RadMatcher** is a self-hosted web app that matches free-text radiology exam
descriptions (the messy strings that arrive on real orders — `ct head w
contrast`, `cr chest 2v`, `mri l-spine wo`) to the structured exam codes in
*your* catalog.

It runs **fully locally**. Matching uses a hybrid of exact mappings,
rule-based scoring, and a machine-learning retrieve-and-rerank model trained
from your own data — **no internet connection or API key is required**. An
optional LLM "second opinion" can be enabled for low-confidence matches, and
works with hosted providers *or* a model running on your own machine.

> ⚠️ **Not a medical device.** RadMatcher is a coding-assistance tool. It does
> not make clinical decisions and its output should be reviewed by a qualified
> person. See [Disclaimer](#disclaimer).

> 📄 Presented at **SIIM 2026** (Society for Imaging Informatics in Medicine
> Annual Meeting) — *AI-Assisted Exam Code Normalization with Continuous
> Self-Training Across Multi-PACS Environments*. See [Background](#background).

---

## Table of contents

- [Background](#background)
- [Features](#features)
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Using the app](#using-the-app)
- [Adding your own code set](#adding-your-own-code-set) ← start here for a new install
- [Generating variations (no LLM needed)](#generating-variations-no-llm-needed)
- [Training the model](#training-the-model)
- [Optional: AI suggestions (LLM fallback)](#optional-ai-suggestions-llm-fallback)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [API reference](#api-reference)
- [Data & privacy](#data--privacy)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Background

RadMatcher is the implementation of the approach presented at the **SIIM 2026
Annual Meeting** (Society for Imaging Informatics in Medicine):

> **AI-Assisted Exam Code Normalization with Continuous Self-Training Across
> Multi-PACS Environments.** Mark Jones, Chief Technology Officer, High Plains
> Radiology. Scientific Research & Applied Informatics Posters & Demonstrations
> (#3015).

Teleradiology groups working across multiple PACS/EMR systems face fragmented
exam-naming conventions and inconsistent procedure codes, which cause workflow
bottlenecks, billing errors, and interoperability gaps. RadMatcher tackles this
with a hybrid of structured parsing (modality, anatomy, contrast, laterality,
view count), statistical similarity (TF-IDF + edit distance), and a supervised
ML reranker trained on validated mappings — plus an optional, human-in-the-loop
LLM self-training loop that proposes matches for low-confidence or novel exams
and folds confirmed corrections back into the training data.

In the production deployment described in the abstract, the system standardized
**40,000+ unique study descriptions** with **>95% automation** and **97% match
accuracy** for structured modalities, cut manual mapping time by **>80%**, and
improved precision **~15%** over a static TF-IDF baseline after six months of
continuous self-training.

---

## Features

- **Free-text → code matching** with confidence scores and ranked alternatives.
- **Bring your own code set** — manage codes in a table UI, or upload a CSV
  (a downloadable template is provided). No hand-editing files required.
- **Automatic variation generation** — expand each code into the many ways
  people actually phrase it (abbreviations, synonyms, modality aliases,
  contrast/laterality phrasing) **without any LLM**.
- **Local ML model** — retrieve-and-rerank trained on your mappings, retrainable
  from the UI with a live progress bar, plus optional nightly retraining.
- **Training Mode** to verify/correct matches and grow your mapping set.
- **Match history** with search and CSV export.
- **Optional AI fallback** for weak matches: OpenAI, Anthropic (Claude),
  Google (Gemini), or any self-hosted OpenAI-compatible server (Ollama, LM
  Studio, vLLM, …). Off by default.
- **REST API** for integrating matching into other systems.

---

## Quick start

```bash
# 1. Get the code
git clone <your-fork-url> radmatcher
cd radmatcher

# 2. Run it. That's it.
python app.py
```

On first launch, `app.py` **automatically**:

1. creates a local virtual environment in `.venv/`,
2. installs the Python dependencies, and
3. builds the ML model from the bundled code set.

The first run can take a few minutes (dependency download + model build).
When you see `Matcher ready`, open:

```
http://localhost:5000
```

Subsequent starts are fast (the model is cached in `matcher_model.pkl`).

> Prefer to manage your own environment? See [Manual setup](#manual-setup).

---

## Requirements

- **Python 3.11+** (required — the pinned numpy/torch stack does not support
  3.10 or older; the app will refuse to start on an older Python).
- ~1–2 GB free disk (ML dependencies: scikit-learn, sentence-transformers,
  lightgbm, torch).
- **macOS only:** lightgbm needs the OpenMP runtime:
  ```bash
  brew install libomp
  ```
- Windows users: the bootstrap automatically uses `requirements-windows.txt`.

The app ships with a working example code set (the `RM` catalog, ~1,360 codes
with Left/Right/Bilateral variants for paired exams) so it runs out of the box.
Replace it with your own catalog at any time — see below.

### Manual setup

If you'd rather not use the auto-bootstrap:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Windows: requirements-windows.txt
python app.py
```

---

## Using the app

1. **Match an exam.** Type a description in the big search box and press
   **Match** (or click an example chip). You'll get the best match with a
   confidence score, plus alternatives.
2. **Manage your codes.** Open **Code Set** in the sidebar to add/edit/delete
   codes, upload a CSV, or generate variations (see below).
3. **Train / correct.** Enter **Training Mode** (sidebar) to verify matches and
   teach the model your corrections.
4. **Review history.** Open **Match History** to search past matches and export
   them.

---

## Adding your own code set

This is the most common task for a new install. A **code** is one row in your
catalog with these fields:

| Column        | Required | Example            | Notes |
|---------------|----------|--------------------|-------|
| `Code`        | ✅       | `CT123`            | Your unique code/ID |
| `description` | ✅       | `CT Head wo IV Contrast` | Human-readable exam name |
| `modality`    | recommended | `CT`            | `CT, MR, XR, US, MG, NM, RF, XA, BMD, PT` |
| `bodyRegion`  | optional | `HEAD`             | Anatomy/region |
| `Laterality`  | optional | `N/A`              | `N/A, LEFT, RIGHT, BILATERAL` |
| `Contrast`    | optional | `WOC`              | `N/A, WC` (with), `WOC` (without), `WWOC` (with & without) |
| `XR#Views`    | optional | `1`                | View count for X-ray |

You have **three ways** to load your codes:

### Option A — Table UI (no CSV)

1. Sidebar → **Code Set**.
2. Click **Add code**, fill in the fields, **Save code**. Repeat.
3. Edit or delete any row inline.

### Option B — CSV upload

1. Sidebar → **Code Set** → **Template** to download `exam_codes_template.csv`.
2. Fill it in (one row per code; the header must include at least `Code` and
   `description`).
3. Back in **Code Set**, choose **Replace all** or **Append new**, then
   **Upload CSV**.

### Option C — Edit the file directly

Replace `exam_codes.csv` in the project folder. The header row must be:

```csv
Code,Laterality,Contrast,XR#Views,description,modality,bodyRegion
```

### After changing codes

Two steps make new codes matchable well:

1. **Generate variations** (below) so the model learns the many phrasings.
2. **Retrain** so the new codes enter the model. A **Retrain now** button
   appears after any code-set change, or use **Retrain Model** in Training Mode.

---

## Generating variations (no LLM needed)

A single code like `MR Lumbar Spine wo IV Contrast` shows up in the wild as
`mri l-spine`, `lumbar mri without contrast`, `ls spine mr`, and dozens more.
RadMatcher can synthesize those variants automatically using a rule-based
generator (`seed_mappings.py`) — **no LLM, no API key, fully offline**. It
expands:

- modality aliases (`CT` ↔ `CAT Scan`, `MR` ↔ `MRI`, `US` ↔ `Ultrasound`, …),
- anatomy synonyms (`Head` ↔ `Brain`, `Kidney` ↔ `Renal`, …),
- contrast phrasing (`wo IV Contrast` ↔ `WOC` ↔ `without contrast`, …),
- laterality and view-count shorthand, spine shorthand, and more.

**To run it:** Sidebar → **Code Set** → **Generate variations**. New mappings
are appended to `exam_mappings.csv` (duplicates are skipped, so it's safe to run
repeatedly). Then **Retrain** to bake them in.

These generated mappings also cross the example-count threshold that turns on
the ML reranker, so even a brand-new catalog gets good matches quickly.

---

## Training the model

The matcher learns from `exam_mappings.csv` (free-text → code pairs):

- **Retrain** from **Code Set** (after changes) or **Training Mode → Retrain
  Model**. Progress shows in the header bar.
- **Training Mode** (password-protected) lets you verify a match as correct or
  pick/enter the right code — each correction is saved as a new mapping.
- **Nightly retraining** can be scheduled in **Settings** (the app must be
  running at the scheduled time).

Set the Training Mode password with the `RADMATCHER_PASSWORD` environment variable
(default: `radmatcher`).

---

## Optional: AI suggestions (LLM fallback)

When the local match is weak, RadMatcher can ask an LLM to suggest a code from
your catalog. **This is optional and off by default** — core matching never
needs it.

Configure it in **Settings → AI Suggestions**:

| Provider | What to enter |
|----------|---------------|
| **OpenAI** | Model (e.g. `gpt-4o`) + API key |
| **Anthropic (Claude)** | Model (e.g. `claude-sonnet-4-5`) + API key |
| **Google (Gemini)** | Model (e.g. `gemini-1.5-flash`) + API key |
| **Self-hosted / OpenAI-compatible** | Model name + **Base URL** (API key usually optional) |

**Self-hosted examples** (run a model on your own machine, nothing leaves it):

- **Ollama** — Base URL `http://localhost:11434/v1`, model e.g. `llama3.1`
- **LM Studio** — Base URL `http://localhost:1234/v1`, model = whatever you loaded
- Also works with vLLM, LocalAI, and other OpenAI-compatible servers, as well as
  gateways like OpenRouter/Together/Groq (set their base URL + key).

The LLM is only ever asked to choose from *your* codes, with guardrails against
hallucinated or wrong-modality answers. Your API key is stored locally in
`app_settings.json` (git-ignored) and never displayed back in the UI.

---

## Configuration

Environment variables (all optional — see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `RADMATCHER_PASSWORD` | `radmatcher` | Training Mode password. **Change this.** |
| `RADMATCHER_SECRET` | random per start | Flask session secret. Set a fixed value to keep logins across restarts. |
| `RADMATCHER_HOST` | `0.0.0.0` | Bind address. |
| `RADMATCHER_PORT` | `5000` | Port. |

App settings (matching thresholds, training options, AI provider) live in
`app_settings.json`, edited through **Settings** in the UI. A reference copy is
in `app_settings.example.json`. This file is git-ignored because it can hold an
API key.

---

## How it works

```
query ──▶ normalize (abbreviations, synonyms, term overrides)
      ──▶ exact mapping?  ─yes─▶ return verified code
      ──▶ candidate retrieval (TF-IDF + embeddings)
      ──▶ rerank (LightGBM) + calibrated confidence
      ──▶ rules/score safety net
      ──▶ (optional) LLM fallback if confidence is low
      ──▶ ranked matches + score breakdown
```

- `exam_codes.csv` is your catalog (source of truth).
- `exam_mappings.csv` are free-text → code examples (seeded variations + your
  verified corrections); they drive both exact matching and model training.
- `matcher_model.pkl` / `reranker.pkl` / `embeddings.npz` are the trained
  artifacts, rebuilt on retrain (not committed to git).
- `match_history.db` is a local SQLite log of recent matches.

---

## Project structure

```
app.py                 Flask app: routes, training orchestration, API
matcher.py             Core matching engine + text normalization
reranker.py            LightGBM reranker + confidence calibration
embeddings.py          Embedding index (sentence-transformers)
seed_mappings.py       Rule-based variation generator (no LLM)
llm_providers.py       Optional multi-provider LLM fallback
llm_review.py          Pending-suggestion review queue
train.py               Standalone training entry point
templates/index.html   Single-page UI (Tailwind via CDN)
exam_codes.csv         Example code catalog (replace with yours)
exam_mappings.csv      Free-text → code training examples
term_replacements.json Regex normalizations applied before matching
requirements*.txt      Python dependencies
```

---

## API reference

Selected endpoints (JSON):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/match` | Match a query. Body: `{"query": "...", "modality": "CT"}` |
| GET  | `/api/codes` | List the full code catalog |
| POST | `/api/codes` | Add a code |
| PUT/DELETE | `/api/codes/<id>` | Edit / delete a code |
| POST | `/api/codes/upload` | Upload a code-set CSV (`mode=replace|append`) |
| GET  | `/api/codes/template` | Download the CSV template |
| POST | `/api/codes/generate_variations` | Generate non-LLM variations |
| POST | `/train` | Start background retraining |
| GET  | `/api/training_status` | Training progress |
| GET  | `/api/match_history` | Recent matches |
| GET  | `/api/health` | Health check |

Example:

```bash
curl -X POST http://localhost:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{"query": "ct abdomen pelvis w contrast"}'
```

> The API has no authentication. Run it on a trusted network, or put it behind
> your own auth/reverse proxy.

---

## Data & privacy

- Matching, training, and history all happen **on your machine**. Nothing is
  sent anywhere unless you explicitly enable a hosted LLM provider in Settings.
- The bundled example codes are RadMatcher's own **`RM` catalog** — an exam set
  with proper Left/Right/Bilateral handling for paired studies. Replace it with
  your own catalog at any time (see [Adding your own code set](#adding-your-own-code-set)).
- Do not commit real patient data. `app_settings.json` (which may contain an API
  key), the SQLite history, and trained model artifacts are git-ignored.

---

## License

**Apache License 2.0 with the Commons Clause** — see [`LICENSE`](LICENSE).

In plain terms:

- ✅ **Free to use, run, modify, and self-host** — including inside a company or
  hospital, for-profit or not, at no cost.
- ✅ Free to share your changes, with a **patent grant** and patent-retaliation
  protection (from Apache 2.0).
- ❌ You may **not sell it** — that includes selling copies, offering it as a
  paid hosted/SaaS product, or charging for hosting/support/consulting whose
  value derives substantially from this software (the Commons Clause).

This keeps RadMatcher free for everyone who wants to use it while preventing
anyone from commercializing it. Because of the no-sale condition it is
"source-available" rather than an OSI-approved open-source license.

---

## Disclaimer

RadMatcher is provided "as is", without warranty of any kind. It is **not** a
medical device and is **not** intended for clinical decision-making. Code
suggestions — including any from an LLM — may be incorrect and must be reviewed
by a qualified person before use. You are responsible for validating results and
for compliance with all applicable regulations and policies (including handling
of any protected health information).
