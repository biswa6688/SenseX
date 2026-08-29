# SenseX — agent context

Read ARCHITECTURE.md and DECISIONS.md before making structural changes. Don't re-litigate decisions logged in DECISIONS.md — they were explicitly chosen by the user, not defaults.

## Fast facts
- Brand: SenseX, AI-motif SVG icon (hand-authored, not sourced externally).
- Three independent apps, NO monorepo tooling: `backend/` (NestJS+TS), `frontend/` (React+TS+Vite), `ml-service/` (Python 3.13 FastAPI). Each has its own package manager state — install/run independently.
- `ml-service` uses `.venv` on **Python 3.13** (not the also-installed 3.14 — no ML wheels for it yet).
- ML inference (STT/TTS/diarization/LLM) lives ONLY in `ml-service`. NestJS never runs models itself, only orchestrates via HTTP.
- Model stack: faster-whisper distil-large-v3 (STT) + pyannote speaker-diarization-3.1 (diarization) + Qwen2.5-3B-Instruct GGUF Q4_K_M via llama-cpp-python (summary/sentiment/QA, one combined call) + Piper (TTS, subprocess). CPU-only, budgeted to fit 16GB RAM — see ARCHITECTURE.md RAM table.
- ml-service job queue is single-worker, concurrency pinned to 1 — never parallelize inference jobs.
- `storage/` at repo root is shared by backend + ml-service (uploads, job results, model cache). Gitignored.
- No Prisma/DB anywhere — job history is read straight off the shared `storage/audio-jobs/*/job.json` sidecars ml-service already writes. Don't reintroduce a DB without a real need (see DECISIONS.md #10).
- Current status (2026-08-29): all three services scaffolded, building, and booting clean. ml-service pipeline (`app/pipeline/*.py`) is fully wired (no stubs) and non-gated models are downloaded (whisper distil-large-v3, Qwen2.5-3B-Instruct GGUF, Piper voice). Diarization needs a user-supplied `HF_TOKEN` (gated model) — falls back to single-speaker without it, doesn't crash. Backend orchestration modules (`storage/`, `ml-client/`, `audio-jobs/`, `models/`, `tts/`) are live and proxy-verified against the real ml-service. Frontend (Tailwind v4, hand-authored components, splash/landing/playground/models/opencode-cli, theme toggle) is live and was verified with a real headed-Playwright run including a full-stack TTS round-trip. See ARCHITECTURE.md "Build order / status" for the detailed checklist and what's still open (chunked long-audio summarization, real end-to-end audio-file test, wavesurfer/Recharts polish, OpenAPI codegen).

## Before touching ml-service heavy deps
ffmpeg is now on PATH (installed via `winget install Gyan.FFmpeg`). llama-cpp-python built from source successfully on this machine (no prebuilt wheel for Python 3.13 yet) — that took several minutes; don't assume it hung if it's slow.
