# IntelliSense — Architecture

Full-stack audio-intelligence app: STT, TTS, Summary, Sentiment, QA Ratings, Transcript, Speaker Diarization. CPU-only, 16GB RAM target machine, best accuracy/speed tradeoff achievable under that constraint.

## Repo layout (no monorepo tooling — plain independent folders)

```
IntelliSence/
  backend/          NestJS + TypeScript, own package.json (npm)
  frontend/         React + TypeScript + Vite, own package.json (npm)
  ml-service/        Python 3.13 FastAPI, own .venv
  storage/            gitignored runtime data, shared by backend + ml-service
    audio-jobs/{jobId}/   original.<ext>, job.json, summary.wav, result.json
    models/{modelId}/     downloaded model weights + .complete marker
    tts/                    standalone free-text TTS outputs
  scripts/            dev launch helpers
```

Explicitly rejected: pnpm/turborepo/nx workspaces, shared-types workspace package, Docker for local dev (see DECISIONS.md). Each app is installed/run independently — `npm install` in backend/ and frontend/ separately, no root install step.

## Why no monorepo tooling
User explicitly rejected it. Three independent runtimes (Node backend, Node frontend, Python ml-service) anyway — a JS-only workspace tool wouldn't span all three, and with only two Node apps the coordination benefit is marginal versus the complexity.

## Services

### ml-service (Python 3.13, FastAPI) — does all ML inference
Runs on **Python 3.13**, not 3.14 (3.14 is too new — no prebuilt CPU wheels yet for torch/ctranslate2/llama-cpp-python as of this writing). Venv at `ml-service/.venv`.

Endpoints:
```
POST /jobs                body {filePath}         -> {jobId, status}   202
GET  /jobs/{jobId}                                 -> {status, stage, error}
GET  /jobs/{jobId}/result                          -> full result JSON
POST /tts                 body {text, voice?}      -> {audioPath}       (standalone free-text TTS)
GET  /models                                       -> list + downloaded flag
POST /models/{id}/download                         -> triggers huggingface_hub download
GET  /models/{id}/status                           -> download progress
GET  /health
```

Single-worker async job queue (`app/core/job_queue.py`), **concurrency pinned to 1** — never run two inference jobs at once. This is what keeps the RAM budget safe, not lazy-loading. Job state kept in memory + persisted as `storage/audio-jobs/{jobId}/job.json` sidecar so it survives a restart.

NestJS writes uploaded audio into shared `storage/`, then calls `POST /jobs` with the **file path**, not re-uploaded bytes — both processes are on the same machine, no need to double-transfer.

### Pipeline (per audio job), see `app/pipeline/`
1. `stt.py` — faster-whisper transcription, `word_timestamps=True`. Model: `distil-large-v3` (English-tuned, ~5.8x faster than large-v3 at ~99% of its accuracy). Swap to `large-v3-turbo` (multilingual) via `WHISPER_MODEL` env var if non-English audio is needed — pure config change.
2. `diarize.py` — pyannote/speaker-diarization-3.1, CPU. **Gated HF model** — requires manual one-time HF account + license acceptance + `HF_TOKEN` in `ml-service/.env`. Cannot be auto-bundled.
3. `merge.py` — **word-level** merge: each word assigned to the diarization turn containing its midpoint, then re-grouped into speaker turns. This (not segment-level overlap) is what gives good speaker-boundary accuracy without a separate forced-alignment model (avoids WhisperX-style extra wav2vec2 weights — no RAM budget for that).
4. `llm_analysis.py` — **one combined LLM call** (not three separate calls) returns `{summary, sentiment, qaRatings}` as one JSON object. Model: Qwen2.5-3B-Instruct, GGUF Q4_K_M, via llama-cpp-python. Combined call avoids re-processing a long transcript prompt 3x — prompt-eval dominates CPU cost. Long calls (>~40-50min) need map-reduce chunking before the final structured pass (not yet implemented — TODO).
5. `tts.py` — Piper, invoked as a **subprocess** (piper CLI binary), not the `piper-tts` pip binding (less stable/consistently packaged historically). Used two ways: (a) pipeline stage synthesizing the generated Summary, (b) standalone `/tts` endpoint for arbitrary free text.

QA Ratings rubric (placeholder, confirm with stakeholder): Greeting, Active Listening, Empathy, Resolution, Compliance, Professionalism — each `{score, rationale}`, plus `overallScore` 1-10.

### RAM budget (16GB ceiling) — why concurrency=1 matters more than lazy-loading
| Component | Est. |
|---|---|
| OS overhead | ~2.0 GB |
| NestJS process | ~0.3 GB |
| Python base (torch/ctranslate2 import overhead) | ~1.5 GB |
| faster-whisper distil-large-v3 int8 | ~1.5 GB |
| pyannote speaker-diarization-3.1 | ~1.5 GB |
| Qwen2.5-3B GGUF Q4_K_M (n_ctx≈8192) | ~3.0 GB |
| Piper | ~0.3 GB |
| **Subtotal, all loaded simultaneously** | **~10.1 GB** |
| Headroom | ~5.9 GB |

Decision: load all four models at startup, keep resident (not lazy-load/unload) — math fits, and lazy-loading adds cold-start-mid-job complexity for marginal benefit. If RAM is tighter on a future host, `MODEL_LOAD_STRATEGY=eager|lazy` is the intended escape hatch (not yet implemented).

### backend (NestJS + TypeScript) — orchestrates, never does inference itself
Modules (planned, not yet scaffolded beyond Nest's default skeleton):
```
audio-jobs/   AudioJobsModule — upload handling (multer/FileInterceptor), writes to storage/, calls ml-service, polls/forwards status
ml-client/    MlClientModule — HttpModule wrapper around ml-service's REST API
models/       ModelsModule — proxies ml-service's /models endpoints
opencode/     OpencodeModule — NOT NEEDED as backend module; OpenCode CLI page is a pure external-link static frontend page (see DECISIONS.md). Keep out of backend scope.
storage/      shared file storage helpers
config/       ConfigModule + env validation
```
Persistence: Prisma + SQLite for an `AudioJob` history table (id, mlJobId, filename, status, stage, error, timestamps, resultJson) — not yet scaffolded. No auth for v1 (local single-user tool).

API surface for frontend (not yet implemented):
```
POST /api/audio-jobs                     multipart upload -> {jobId, status}
GET  /api/audio-jobs/:id                 status/stage
GET  /api/audio-jobs/:id/result          combined result DTO
GET  /api/audio-jobs/:id/audio/original  stream original upload
GET  /api/audio-jobs/:id/audio/summary   stream generated summary TTS wav
POST /api/tts                            standalone free-text TTS -> stream wav
GET  /api/models                         proxies ml-service
POST /api/models/:id/download
GET  /api/models/:id/status
```
No backend endpoint for OpenCode CLI — frontend links directly to the official upstream release page.

### frontend (React + TypeScript + Vite)
Stack: `react-router-dom`, Tailwind CSS + shadcn/ui (CSS-variable theming satisfies dark/light/system directly), TanStack Query (job-status polling), Zustand (theme/active-job state), Framer Motion (splash/progress animation), Recharts (sentiment/QA charts), wavesurfer.js (waveform + diarization-colored regions), react-dropzone + MediaRecorder (`useAudioRecorder` hook) for upload/record.

Routes:
```
/            SplashGate wraps app; shows brand splash 5s (sessionStorage-gated, once per browser session), then Landing
/playground  upload/record -> submit -> poll -> tabbed results (Transcript, Diarization, Summary+TTS playback, Sentiment, QA Ratings) + standalone free-text TTS tool
/models      model list + per-model download progress + HF_TOKEN setup instructions for diarization
/opencode-cli  static page, links to official OpenCode CLI GitHub releases per platform — no backend calls
```
Brand icon: hand-authored inline SVG (abstract neural-node/interconnected-dot AI motif), `currentColor`-based for theme adaptation, not sourced externally.

## Shared types
No workspace package. NestJS DTOs (via `@nestjs/swagger`) are the source of truth. Frontend generates a typed client from Nest's `/api-json` via `openapi-typescript` + `openapi-fetch` (script TBD, e.g. `frontend/scripts/generate-api-types.mjs` run manually or via npm script — no workspace linking needed, it's just a fetch-and-write codegen step). Two small stable enums (`JobStatus`, `PipelineStage`) may be hand-duplicated in both apps rather than generated — not worth codegen ceremony.

## Build order / status
1. ✅ Repo scaffold — backend (Nest), frontend (Vite React-TS), ml-service (FastAPI skeleton, Python 3.13 venv).
2. ✅ ML pipeline wired — all heavy deps installed and import-clean (torch 2.13+cpu, faster-whisper, ctranslate2, pyannote.audio 4.0.7, llama-cpp-python 0.3.35, piper-tts). `app/pipeline/*.py` has real implementations (stt, diarize-with-fallback, merge, llm_analysis, tts), no more `NotImplementedError` stubs. Non-gated models downloaded and cached: whisper distil-large-v3, Qwen2.5-3B-Instruct GGUF Q4_K_M, Piper en_US-lessac-medium voice. Diarization model still needs a user-provided `HF_TOKEN` (gated) — falls back to single-speaker until then, does not crash the pipeline. **Not yet done**: a real end-to-end run on an actual audio file to validate accuracy and measure real peak RAM against the ~10GB budget estimate.
3. ✅ FastAPI job queue wired to the real pipeline (`app/routers/jobs.py`), single-worker concurrency=1, persists `job.json` + `result.json` per job.
4. ✅ NestJS orchestration modules — `storage/`, `ml-client/`, `audio-jobs/`, `models/`, `tts/` (standalone free-text TTS, per DECISIONS.md #4). No Prisma/DB — history reads the shared filesystem directly (DECISIONS.md #10). Builds clean, boots clean, proxies verified live end-to-end (`GET /api/models` → real ml-service data).
5. ⚠️ Partial: TanStack Query wired throughout the frontend; OpenAPI codegen NOT done — hand-written client in `frontend/src/api/client.ts` instead (DECISIONS.md #11). Still a backlog item if drift becomes a problem.
6. ✅ Functional Playground — real upload/record→submit→poll→tabbed results (Transcript/Diarization/Summary+TTS/Sentiment/QA), plus the standalone free-text TTS tool. Verified with a live Playwright (headed, real Chromium) run: splash→landing→playground→theme toggle→models→opencode-cli navigation, and a real end-to-end TTS call through frontend→backend→ml-service→Piper.
7. ✅ Design system + visual polish (first pass) — Tailwind v4 (CSS-first, `@tailwindcss/vite`), hand-authored components (no shadcn CLI used), light/dark/system theming via CSS custom properties + a `.dark` class toggled by a Zustand store, Framer Motion splash (5s, session-gated) and landing animations, hand-authored brand SVG (`components/Logo.tsx`, also becomes `public/favicon.svg`). **Backlog**: wavesurfer.js waveform view and Recharts-based charts from the original design were skipped for time — current diarization/QA visuals are plain CSS bars, functional but plainer than originally planned.
8. ✅ Models page (live download status + progress trigger, HF token guidance for diarization) + OpenCode CLI static page (external links only, per DECISIONS.md #5).
9. ⬜ Hardening — no chunked long-audio map-reduce summarization yet (noted TODO in `llm_analysis.py`), no upload size/type validation, no a11y pass, no automated test suite (only the one manual Playwright smoke script, which was run once and deleted — not kept in the repo).

### Known-good verification (2026-08-29)
- `ml-service`: `python -c "import app.main"` clean; `uvicorn` boots; `GET /health` → `{"status":"ok"}`.
- `backend`: `npm run build` clean; `npm run start:dev` boots, all routes mapped; `GET /api/models` proxies live ml-service data.
- `frontend`: `tsc --noEmit` clean; `npm run dev` boots.
- **Full pipeline run end-to-end on a real synthesized 9-turn call recording** (via `/api/audio-jobs`): transcript accurate, summary/sentiment/QA all sensible with rationale, diarization correctly fell back to single-speaker (no `HF_TOKEN` set). ~74s wall time for ~75s of audio — roughly real-time on CPU. Peak RAM with whisper + the 3B LLM resident: **4.57GB working set** — comfortably under the ~10GB budget estimate even before adding pyannote's ~1.5GB (not loaded in this run).

### Bugs found and fixed during that first real run
Worth noting because they'd all have been invisible without an actual end-to-end run — the earlier "verified" Playwright pass was a false positive (see below):
1. **Event loop blocking (serious)**: `job_queue._worker()` awaited pipeline handlers that were `async def` in name only — no real `await` inside, just synchronous CPU-bound calls (whisper/pyannote/llama.cpp). This froze FastAPI's single-threaded event loop for the entire job duration, including the response to the very request that started the job (`POST /jobs` took 22s instead of being instant) and all status polls. Fixed: handlers are now plain synchronous functions, run via `asyncio.to_thread` in the worker (`app/core/job_queue.py`, `app/routers/jobs.py`). The standalone `/tts` endpoint had the same issue on a smaller scale, fixed via `run_in_threadpool` (`app/routers/tts.py`).
2. **Piper voice path mismatch**: `tts.py` assumed a flat `{voice}.onnx` filename, but `hf_hub_download` preserves the HF repo's nested path (`en/en_US/lessac/medium/en_US-lessac-medium.onnx`). Fixed by sharing one `PIPER_VOICE_REPO_PATHS` mapping between `tts.py` (now the source of truth) and `routers/models.py` (imports it) instead of two independent copies.
3. **LLM model path double-prefixed**: `config.py`'s default `llm_model_path` was `"models/qwen2.5-3b-instruct-q4_k_m.gguf"`, joined onto `models_cache_dir` (already `.../storage/models`) — produced `storage/models/models/...`, which doesn't exist; the real download target is `storage/models/llm/...`. Fixed the default to `"llm/qwen2.5-3b-instruct-q4_k_m.gguf"`.
4. **Frontend swallowed backend errors as fake success**: `api/client.ts`'s `speak()` called `res.blob()` without checking `res.ok` first, so a failed request still resolved as a "successful" mutation with a garbage blob, and the Playwright check `waitForSelector('audio')` passed because the element existed — regardless of whether it held real audio. This is why the earlier Playwright pass didn't catch bug #1/#2 above. Fixed to throw on non-OK responses like the rest of the client. **Lesson**: a DOM-presence check is not a correctness check for anything backed by a network call — assert on response status or actual content, not element existence.

### Full re-verification after the fixes above (still 2026-08-29)
Re-ran through the actual React UI (not curl) with content-level assertions this time:
- Free-text TTS: clicked Speak in the real UI, fetched the resulting `<audio src="blob:...">`'s bytes directly in-browser — 133KB, real WAV.
- Full audio-job pipeline: uploaded the same 9-turn test call through the real drag-and-drop UI, waited for actual completion, and read real page content — Transcript tab shows the correct transcribed text, Summary tab has a working audio player, QA Ratings tab shows a real scored rubric. All passed.
- One test-script-only bug surfaced and was fixed in the process (not an app bug): an early version of the verification script used a loose `text=Transcript` selector that matched the landing paragraph's copy ("...pipeline: transcript, diarization...") and resolved before the pipeline had actually finished. Fixed by targeting the actual tab button via `getByRole('button', { name: ... })`. Worth remembering when writing future browser tests here — prefer role/exact selectors over loose text substrings.

## Known risks / unresolved
- **Python 3.14 vs 3.13**: only these two are installed on this machine; using 3.13 for ml-service. Re-check wheel availability if bumping later.
- **ffmpeg not on PATH**: required for decoding non-WAV uploads (webm/opus from browser recording, mp3/m4a uploads). Must be installed manually (winget/choco) before Phase 2 spike.
- **llama-cpp-python on Windows**: may need to build from source (MSVC Build Tools) if no prebuilt wheel matches Python 3.13 — validate early in Phase 2.
- **pyannote HF_TOKEN**: manual per-user setup, cannot be automated — Models page must surface this clearly, not fail silently.
- **QA rubric criteria**: placeholder list in `models.py`/`llm_analysis.py`, needs stakeholder confirmation.
- **Max audio length / chunked summarization**: designed for but not yet implemented.
