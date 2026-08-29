# SenseX

Audio intelligence app: Speech-to-Text, Text-to-Speech, Summary, Sentiment, QA Ratings, full Transcript, Speaker Diarization. CPU-only, tuned to run on 16GB RAM.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design and [DECISIONS.md](DECISIONS.md) for why things are built this way.

## Structure

Three independent apps, no monorepo tooling — install/run each separately.

```
backend/       NestJS + TypeScript API (orchestration only, no ML inference)
frontend/      React + TypeScript + Vite UI
ml-service/    Python 3.13 FastAPI — all ML inference (STT/TTS/diarization/LLM)
storage/       shared runtime data (uploads, job results, model cache) — gitignored
```

## Setup

### Prerequisites
- Node.js 20+, npm
- Python 3.13 (not 3.14 — no ML wheels for it yet)
- ffmpeg on PATH (required for decoding non-WAV audio uploads) — not yet installed on this machine, install via winget/choco first
- A free HuggingFace account + accepted license for `pyannote/speaker-diarization-community-1`, with a token set as `HF_TOKEN` in `ml-service/.env` (required for diarization; cannot be automated)

### Backend
```
cd backend
npm install
npm run start:dev
```

### Frontend
```
cd frontend
npm install
npm run dev
```

### ML service
```
cd ml-service
.venv\Scripts\activate
pip install -r requirements.txt      # lightweight deps only; heavy ML deps are commented out, see file
uvicorn app.main:app --reload --port 8000
```
Heavy ML deps (torch, faster-whisper, ctranslate2, pyannote.audio, llama-cpp-python, huggingface_hub) are intentionally commented out in `requirements.txt` pending the Phase 2 pipeline spike — install torch with `--index-url https://download.pytorch.org/whl/cpu` to avoid pulling CUDA wheels.

## Status
All three services build/boot clean and are wired end-to-end (verified with a live headed-Playwright run, including a real TTS round-trip through the full stack). Non-gated models are downloaded. Diarization needs a one-time `HF_TOKEN` setup (see Prerequisites) — until then it falls back to single-speaker rather than failing. See ARCHITECTURE.md build-order checklist for what's left (chunked long-audio summarization, a real end-to-end audio-file run, visual polish).
